//! Optional NVIDIA GPU backend.
//!
//! The GPU code is compiled only with the cargo feature `cuda`; without it
//! every function here still exists but reports that the build has no CUDA
//! support, so the Python API is the same on every machine. With the feature,
//! the NVIDIA driver and the CUDA runtime compiler (NVRTC) are loaded at run
//! time, so a build with `cuda` also runs (on the CPU) on a machine that has
//! no NVIDIA card at all.

use super::{column_f64, column_i64, heap_descriptor, random_table, Algo, FitsHeader, FitsStats, HduSelector, locate_image, open_mmap, stats_impl};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;

/// Which device computes the statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Cpu,
    Gpu,
    /// Use the GPU when there is a usable NVIDIA one and the work suits it,
    /// the CPU otherwise.
    Auto,
}

/// What `fits_stats` does when neither `device=` nor `EUCLIDRUST_DEVICE` is given.
const DEFAULT_DEVICE: Device = Device::Auto;
/// With `device="auto"` the GPU is used for Rice-compressed images (the compressed data is five
/// times smaller to send over the PCIe bus, and decoding is heavy), as long as there are enough
/// tiles to keep the card busy (one thread decodes one tile) and the file is big enough to
/// pay for starting CUDA (about 0.2 s, once per process). Raw images always stay on the CPU:
/// on the machines measured, sending them over the bus costs more than computing on the CPU.
const AUTO_MIN_TILES: usize = 256;
const AUTO_MIN_FILE_BYTES: usize = 64 << 20;

impl Device {
    /// `arg` is the `device=` argument; when it is absent the environment
    /// variable `EUCLIDRUST_DEVICE` is used, then the built-in default.
    pub fn parse(arg: Option<&str>) -> PyResult<Device> {
        let text = arg.map(|s| s.to_string()).or_else(|| std::env::var("EUCLIDRUST_DEVICE").ok());
        match text.as_deref().map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") => Ok(DEFAULT_DEVICE),
            Some("cpu") => Ok(Device::Cpu),
            Some("gpu") | Some("cuda") => Ok(Device::Gpu),
            Some("auto") => Ok(Device::Auto),
            Some(other) => Err(PyValueError::new_err(format!(
                "unknown device '{other}': use 'cpu', 'gpu' or 'auto'"
            ))),
        }
    }
}

/// What the machine offers, as reported by `gpu_info()`.
#[pyclass]
pub struct GpuInfo {
    #[pyo3(get)]
    pub available: bool, // an NVIDIA GPU, its driver and the CUDA compiler are all usable
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub compute_capability: String,
    #[pyo3(get)]
    pub memory_total_mb: u64,
    #[pyo3(get)]
    pub memory_free_mb: u64,
    #[pyo3(get)]
    pub message: String, // why it is not available, or a short description
}

#[pymethods]
impl GpuInfo {
    fn __repr__(&self) -> String {
        if self.available {
            format!(
                "<GpuInfo available=yes name={:?} compute_capability={} memory={} MB ({} MB free)>",
                self.name, self.compute_capability, self.memory_total_mb, self.memory_free_mb
            )
        } else {
            format!("<GpuInfo available=no ({})>", self.message)
        }
    }
}

/// Everything the GPU needs to decode the tiles of a Rice-compressed image.
pub struct RiceJob {
    pub heap_file_offset: u64, // where the compressed data (the table's heap) starts in the file
    pub heap_len: usize,
    pub offsets: Vec<u64>, // per tile: where its bytes start in the heap
    pub lens: Vec<u32>,
    pub npix: Vec<u32>,
    pub scale: Vec<f64>,
    pub zero: Vec<f64>,
    pub tnull: Vec<i64>,
    pub blocksize: i32,
    pub bytepix: i32,
    pub dither: i32,
    pub dither_seed: i64,
    pub is_float: bool,
    pub f32_out: bool,
    pub bzero: f64,
    pub bscale: f64,
}

/// Raw result of a reduction on the GPU.
pub struct Raw {
    pub min: f64,
    pub max: f64,
    pub sum: f64,
    pub nans: u64,
    pub upload_ms: Option<f64>,
    pub kernel_ms: Option<f64>,
}

#[cfg(feature = "cuda")]
mod backend {
    use super::RiceJob;
    use super::{GpuInfo, Raw};
    use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PinnedHostSlice, PushKernelArg};
    use rayon::prelude::*;
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    /// The GPU kernels (statistics of raw pixels, and Rice decoding), in `kernels.cu`. They are
    /// shipped inside the binary already compiled to PTX (`kernels.ptx`, made by
    /// `scripts/build_ptx.py`), which the NVIDIA driver finishes for the card in use: only
    /// the driver is needed, not the CUDA toolkit. If the driver refuses the PTX (a driver
    /// older than the one that made it), the source is compiled with the toolkit's NVRTC.
    const SOURCE: &str = include_str!("kernels.cu");
    const EMBEDDED_PTX: &str = include_str!("kernels.ptx");

    const BLOCK: u32 = 256;
    const GRID: u32 = 1024;
    /// Size of one staging buffer in page-locked host memory (two are used, in turn).
    const CHUNK_BYTES: usize = 64 << 20;

    struct Gpu {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        stats_fn: CudaFunction,
        rice_fn: CudaFunction,
        name: String,
        cc: (i32, i32),
        total_mem: usize,
        kernels: &'static str, // where the GPU kernels came from
        /// Two page-locked staging buffers and the device buffer they are copied to,
        /// allocated on first use and kept (freeing page-locked memory takes ~100 ms).
        staging: Vec<PinnedHostSlice<u8>>,
        dev_chunk: Option<CudaSlice<u8>>,
    }

    static GPU: OnceLock<Result<Mutex<Gpu>, String>> = OnceLock::new();

    /// An image copied to the GPU, kept there for repeated use.
    pub struct Resident {
        buf: CudaSlice<u8>,
        pub len: usize,
        pub upload_ms: f64,
    }

    /// The CUDA toolkit's DLLs (NVRTC) are not on the default search path on Windows.
    #[cfg(windows)]
    fn add_cuda_dll_dir() {
        unsafe extern "system" {
            fn SetDllDirectoryW(path: *const u16) -> i32;
        }
        if let Ok(root) = std::env::var("CUDA_PATH") {
            for sub in ["bin\\x64", "bin"] {
                let dir = std::path::Path::new(&root).join(sub);
                if dir.is_dir() {
                    let wide: Vec<u16> = dir.to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
                    unsafe { SetDllDirectoryW(wide.as_ptr()) };
                    return;
                }
            }
        }
    }
    #[cfg(not(windows))]
    fn add_cuda_dll_dir() {}

    /// Compiles the kernel source with the CUDA toolkit's NVRTC (to PTX text).
    fn compile_with_nvrtc() -> Result<Ptx, String> {
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err("the CUDA toolkit's compiler (NVRTC) was not found".to_string());
        }
        compile_ptx_with_opts(SOURCE, CompileOptions { arch: Some("compute_75"), fmad: Some(false), ..Default::default() })
            .map_err(|e| format!("CUDA kernel compilation failed: {e}"))
    }

    /// PTX text of the kernels, compiled now by NVRTC: used by `scripts/build_ptx.py`.
    pub fn compile_ptx_text() -> Result<String, String> {
        add_cuda_dll_dir();
        Ok(compile_with_nvrtc()?.to_src())
    }

    fn load_kernels(ctx: &Arc<CudaContext>) -> Result<(Arc<CudaModule>, &'static str), String> {
        match ctx.load_module(Ptx::from_src(EMBEDDED_PTX)) {
            Ok(m) => Ok((m, "precompiled kernels built into the library")),
            Err(first) => {
                let ptx = compile_with_nvrtc().map_err(|e| {
                    format!("the NVIDIA driver cannot load the library's built-in GPU kernels ({first}) and they cannot be rebuilt: {e}")
                })?;
                let m = ctx.load_module(ptx).map_err(|e| format!("cannot load the CUDA kernels: {e}"))?;
                Ok((m, "compiled at run time with the CUDA toolkit's NVRTC"))
            }
        }
    }

    fn init() -> Result<Mutex<Gpu>, String> {
        add_cuda_dll_dir();
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("no NVIDIA driver found (nvcuda.dll / libcuda.so is missing)".to_string());
        }
        let built = std::panic::catch_unwind(|| -> Result<Gpu, String> {
            let ctx = CudaContext::new(0).map_err(|e| format!("cannot initialise CUDA device 0: {e}"))?;
            let name = ctx.name().map_err(|e| e.to_string())?;
            let cc = ctx.compute_capability().map_err(|e| e.to_string())?;
            let total_mem = ctx.total_mem().map_err(|e| e.to_string())?;
            let (module, kernels) = load_kernels(&ctx)?;
            let stats_fn = module.load_function("stats_kernel").map_err(|e| e.to_string())?;
            let rice_fn = module.load_function("rice_kernel").map_err(|e| e.to_string())?;
            let stream = ctx.default_stream();
            Ok(Gpu { ctx, stream, stats_fn, rice_fn, name, cc, total_mem, kernels, staging: Vec::new(), dev_chunk: None })
        });
        match built {
            Ok(Ok(g)) => Ok(Mutex::new(g)),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("the CUDA libraries could not be loaded".to_string()),
        }
    }

    fn gpu() -> Result<&'static Mutex<Gpu>, String> {
        GPU.get_or_init(init).as_ref().map_err(|e| e.clone())
    }

    pub fn probe() -> GpuInfo {
        match gpu() {
            Ok(m) => {
                let g = m.lock().unwrap();
                let free = g.ctx.mem_get_info().map(|(f, _)| f).unwrap_or(0);
                GpuInfo {
                    available: true,
                    name: g.name.clone(),
                    compute_capability: format!("{}.{}", g.cc.0, g.cc.1),
                    memory_total_mb: (g.total_mem >> 20) as u64,
                    memory_free_mb: (free >> 20) as u64,
                    message: format!("an NVIDIA GPU is usable (kernels: {})", g.kernels),
                }
            }
            Err(e) => GpuInfo {
                available: false,
                name: String::new(),
                compute_capability: String::new(),
                memory_total_mb: 0,
                memory_free_mb: 0,
                message: e,
            },
        }
    }

    pub fn device_name() -> String {
        gpu().map(|m| m.lock().unwrap().name.clone()).unwrap_or_default()
    }

    /// Adds the partial results of the blocks.
    fn combine(partials: &[f64], acc: &mut (f64, f64, f64, u64)) {
        for p in partials.chunks_exact(4) {
            acc.0 = acc.0.min(p[0]);
            acc.1 = acc.1.max(p[1]);
            acc.2 += p[2];
            acc.3 += p[3] as u64;
        }
    }

    fn launch(g: &Gpu, buf: &CudaSlice<u8>, len: usize, bitpix: i32, bzero: f64, bscale: f64, out: &mut CudaViewMut<'_, f64>) -> Result<(), String> {
        let bytes_per = (bitpix.unsigned_abs() as usize) / 8;
        let elems = (len / bytes_per) as u64;
        let view = buf.slice(0..len);
        let cfg = LaunchConfig { grid_dim: (GRID, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        let mut b = g.stream.launch_builder(&g.stats_fn);
        b.arg(&view).arg(&elems).arg(&bitpix).arg(&bzero).arg(&bscale).arg(out);
        unsafe { b.launch(cfg) }.map_err(|e| format!("kernel launch failed: {e}"))?;
        Ok(())
    }

    /// Fills `dst` with `dst.len()` bytes of the file at `offset`, using all the CPU threads:
    /// each maps its own small window of the file, copies it into the page-locked buffer and
    /// releases it (one big mapping would be slow to tear down on Windows).
    fn fill_from_file(dst: &mut [u8], path: &str, offset: u64) -> Result<(), String> {
        const PART: usize = 8 << 20;
        dst.par_chunks_mut(PART)
            .enumerate()
            .try_for_each_init(
                || std::fs::File::open(path),
                |file, (i, part)| {
                    let file = file.as_ref().map_err(|e| format!("cannot open {path}: {e}"))?;
                    let len = part.len();
                    crate::with_window(file, offset + (i * PART) as u64, len, |window| part.copy_from_slice(window))
                        .map_err(|e| format!("cannot read {path}: {e}"))
                },
            )
    }

    /// Makes sure the two staging buffers and the device chunk exist.
    fn ensure_buffers(g: &mut Gpu) -> Result<(), String> {
        if g.staging.is_empty() {
            for _ in 0..2 {
                let buf = unsafe { g.ctx.alloc_pinned_with_flags::<u8>(CHUNK_BYTES, cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED) }
                    .map_err(|e| format!("cannot allocate page-locked memory: {e}"))?;
                g.staging.push(buf);
            }
        }
        if g.dev_chunk.is_none() {
            g.dev_chunk = Some(
                g.stream.alloc_zeros::<u8>(CHUNK_BYTES).map_err(|e| format!("cannot allocate {CHUNK_BYTES} bytes on the GPU: {e}"))?,
            );
        }
        Ok(())
    }

    /// Statistics of `len` bytes of a file. The data goes to the card through two page-locked
    /// staging buffers, used in turn: while the card copies one (and reduces it), all the CPU
    /// threads read the next piece of the file into the other.
    pub fn stats_chunked(path: &str, offset: u64, len_total: usize, bitpix: i32, bzero: f64, bscale: f64) -> Result<Raw, String> {
        let mut guard = gpu()?.lock().unwrap();
        let g: &mut Gpu = &mut guard;
        let debug = std::env::var_os("EUCLIDRUST_GPU_DEBUG").is_some();
        let t_start = Instant::now();
        let (mut t_wait, mut t_fill, mut t_enqueue) = (0.0f64, 0.0f64, 0.0f64);
        let chunk = CHUNK_BYTES;
        let n_chunks = len_total.div_ceil(chunk).max(1);
        ensure_buffers(g)?;
        let mut out = g.stream.alloc_zeros::<f64>(GRID as usize * 4 * n_chunks).map_err(|e| e.to_string())?;
        let t_alloc = t_start.elapsed().as_secs_f64() * 1000.0;
        // `done[i]` marks the moment the card has finished with staging buffer i
        let mut done: Vec<Option<CudaEvent>> = vec![None, None];
        for k in 0..n_chunks {
            let off = k * chunk;
            let len = chunk.min(len_total - off);
            let which = k % 2;
            let t0 = Instant::now();
            if let Some(ev) = done[which].take() {
                ev.synchronize().map_err(|e| e.to_string())?; // the card is done reading this buffer
            }
            t_wait += t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = Instant::now();
            {
                let dst = g.staging[which].as_mut_slice().map_err(|e| e.to_string())?;
                fill_from_file(&mut dst[..len], path, offset + off as u64)?;
            }
            t_fill += t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = Instant::now();
            {
                let dev = g.dev_chunk.as_mut().unwrap();
                let mut view = dev.slice_mut(0..chunk);
                g.stream.memcpy_htod(&g.staging[which], &mut view).map_err(|e| format!("copy to the GPU failed: {e}"))?;
            }
            let mut part = out.slice_mut(k * GRID as usize * 4..(k + 1) * GRID as usize * 4);
            launch(g, g.dev_chunk.as_ref().unwrap(), len, bitpix, bzero, bscale, &mut part)?;
            done[which] = Some(g.stream.record_event(None).map_err(|e| e.to_string())?);
            t_enqueue += t0.elapsed().as_secs_f64() * 1000.0;
        }
        let t0 = Instant::now();
        g.stream.synchronize().map_err(|e| e.to_string())?;
        let t_final = t0.elapsed().as_secs_f64() * 1000.0;
        let partials: Vec<f64> = g.stream.clone_dtoh(&out).map_err(|e| e.to_string())?;
        if debug {
            eprintln!(
                "[gpu] {n_chunks} chunks of {} MB | allocation {t_alloc:.1} ms | waiting for the card {t_wait:.1} ms | reading the file into staging buffers {t_fill:.1} ms | enqueue copy+kernel {t_enqueue:.1} ms | final wait {t_final:.1} ms | total {:.1} ms",
                chunk >> 20,
                t_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let mut acc = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64, 0u64);
        combine(&partials, &mut acc);
        Ok(Raw { min: acc.0, max: acc.1, sum: acc.2, nans: acc.3, upload_ms: None, kernel_ms: None })
    }

    /// Copies `len` bytes of the file to the card and keeps them there.
    pub fn upload(path: &str, offset: u64, len: usize) -> Result<Resident, String> {
        let mut guard = gpu()?.lock().unwrap();
        let g: &mut Gpu = &mut guard;
        let t0 = Instant::now();
        let chunk = CHUNK_BYTES;
        let n_chunks = len.div_ceil(chunk).max(1);
        ensure_buffers(g)?;
        let mut buf = g
            .stream
            .alloc_zeros::<u8>(n_chunks * chunk)
            .map_err(|e| format!("cannot allocate {} bytes on the GPU: {e}", n_chunks * chunk))?;
        let mut done: Vec<Option<CudaEvent>> = vec![None, None];
        for k in 0..n_chunks {
            let off = k * chunk;
            let piece = chunk.min(len - off);
            let which = k % 2;
            if let Some(ev) = done[which].take() {
                ev.synchronize().map_err(|e| e.to_string())?;
            }
            {
                let dst = g.staging[which].as_mut_slice().map_err(|e| e.to_string())?;
                fill_from_file(&mut dst[..piece], path, offset + off as u64)?;
            }
            let mut view = buf.slice_mut(off..off + chunk);
            g.stream.memcpy_htod(&g.staging[which], &mut view).map_err(|e| format!("copy to the GPU failed: {e}"))?;
            done[which] = Some(g.stream.record_event(None).map_err(|e| e.to_string())?);
        }
        g.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(Resident { buf, len, upload_ms: t0.elapsed().as_secs_f64() * 1000.0 })
    }

    /// Statistics of a Rice-compressed image: the compressed data is copied to the card, and one
    /// GPU thread decodes each tile and accumulates its statistics.
    pub fn stats_rice(path: &str, job: &RiceJob) -> Result<Raw, String> {
        let heap = upload(path, job.heap_file_offset, job.heap_len)?;
        let mut guard = gpu()?.lock().unwrap();
        let g: &mut Gpu = &mut guard;
        let n_tiles = job.offsets.len();
        let s = &g.stream;
        let e = |x: cudarc::driver::DriverError| x.to_string();
        let d_off = s.clone_htod(&job.offsets).map_err(e)?;
        let d_len = s.clone_htod(&job.lens).map_err(e)?;
        let d_npix = s.clone_htod(&job.npix).map_err(e)?;
        let d_scale = s.clone_htod(&job.scale).map_err(e)?;
        let d_zero = s.clone_htod(&job.zero).map_err(e)?;
        let d_tnull = s.clone_htod(&job.tnull).map_err(e)?;
        let d_rnd = s.clone_htod(super::random_table()).map_err(e)?;
        let mut out = s.alloc_zeros::<f64>(n_tiles * 4).map_err(e)?;
        let (n_tiles_i, blocksize, bytepix, dither) = (n_tiles as i32, job.blocksize, job.bytepix, job.dither);
        let (is_float, f32_out) = (job.is_float as i32, job.f32_out as i32);
        let cfg = LaunchConfig { grid_dim: (n_tiles.div_ceil(128) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let mut b = s.launch_builder(&g.rice_fn);
        b.arg(&heap.buf).arg(&d_off).arg(&d_len).arg(&d_npix).arg(&d_scale).arg(&d_zero).arg(&d_tnull);
        b.arg(&n_tiles_i).arg(&blocksize).arg(&bytepix).arg(&dither).arg(&job.dither_seed).arg(&is_float).arg(&f32_out);
        b.arg(&job.bzero).arg(&job.bscale).arg(&d_rnd).arg(&mut out);
        unsafe { b.launch(cfg) }.map_err(|e| format!("kernel launch failed: {e}"))?;
        s.synchronize().map_err(e)?;
        let partials: Vec<f64> = s.clone_dtoh(&out).map_err(e)?;
        if partials.chunks_exact(4).any(|p| p[3] > 1e29) {
            return Err("a Rice-compressed tile is corrupt (it ends before all its pixels are decoded)".to_string());
        }
        let mut acc = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64, 0u64);
        combine(&partials, &mut acc);
        Ok(Raw { min: acc.0, max: acc.1, sum: acc.2, nans: acc.3, upload_ms: None, kernel_ms: None })
    }

    /// Statistics of an image that is already on the card.
    pub fn stats_resident(res: &Resident, bitpix: i32, bzero: f64, bscale: f64) -> Result<Raw, String> {
        let g = gpu()?.lock().unwrap();
        let mut out = g.stream.alloc_zeros::<f64>(GRID as usize * 4).map_err(|e| e.to_string())?;
        let t1 = Instant::now();
        {
            let mut whole = out.slice_mut(..);
            launch(&g, &res.buf, res.len, bitpix, bzero, bscale, &mut whole)?;
        }
        g.stream.synchronize().map_err(|e| e.to_string())?;
        let partials: Vec<f64> = g.stream.clone_dtoh(&out).map_err(|e| e.to_string())?;
        let kernel = t1.elapsed().as_secs_f64() * 1000.0;
        let mut acc = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64, 0u64);
        combine(&partials, &mut acc);
        Ok(Raw { min: acc.0, max: acc.1, sum: acc.2, nans: acc.3, upload_ms: None, kernel_ms: Some(kernel) })
    }
}

#[cfg(not(feature = "cuda"))]
mod backend {
    use super::{GpuInfo, Raw};

    const MSG: &str = "this build of euclidrust has no CUDA support (rebuild with: maturin develop --release --features cuda)";

    pub struct Resident {
        pub len: usize,
        pub upload_ms: f64,
    }

    pub fn probe() -> GpuInfo {
        GpuInfo {
            available: false,
            name: String::new(),
            compute_capability: String::new(),
            memory_total_mb: 0,
            memory_free_mb: 0,
            message: MSG.to_string(),
        }
    }
    pub fn device_name() -> String {
        String::new()
    }
    pub fn compile_ptx_text() -> Result<String, String> {
        Err(MSG.to_string())
    }
    pub fn stats_chunked(_: &str, _: u64, _: usize, _: i32, _: f64, _: f64) -> Result<Raw, String> {
        Err(MSG.to_string())
    }
    pub fn upload(_: &str, _: u64, _: usize) -> Result<Resident, String> {
        Err(MSG.to_string())
    }
    pub fn stats_resident(_: &Resident, _: i32, _: f64, _: f64) -> Result<Raw, String> {
        Err(MSG.to_string())
    }
    pub fn stats_rice(_: &str, _: &super::RiceJob) -> Result<Raw, String> {
        Err(MSG.to_string())
    }
}

/// Compiles the GPU kernels to PTX text with the CUDA toolkit; used to regenerate `kernels.ptx`.
#[pyfunction]
pub fn _compile_kernels_ptx(python: Python<'_>) -> PyResult<String> {
    python.detach(backend::compile_ptx_text).map_err(PyValueError::new_err)
}

/// Reports whether an NVIDIA GPU can be used, and which one.
#[pyfunction]
pub fn gpu_info(python: Python<'_>) -> GpuInfo {
    python.detach(backend::probe)
}

/// True when an NVIDIA GPU, its driver and the CUDA runtime compiler are all usable.
#[pyfunction]
pub fn gpu_available(python: Python<'_>) -> bool {
    python.detach(|| backend::probe().available)
}

/// Builds the description of a Rice-compressed image for the GPU, or says why it cannot be done.
fn rice_job(mmap: &[u8], header: &FitsHeader) -> Result<RiceJob, String> {
    let t = header.tiled.as_ref().ok_or("not a tile-compressed image")?;
    if t.algo != Algo::Rice {
        return Err("only Rice-compressed (RICE_1) images are handled on the GPU".to_string());
    }
    let comp = t.comp.ok_or("the table has no COMPRESSED_DATA column")?;
    let is_float = t.zbitpix < 0;
    if is_float && (t.zscale.is_none() || t.zzero.is_none()) {
        return Err("lossless floating-point tiles are not handled on the GPU yet".to_string());
    }
    if !is_float && !matches!(t.zbitpix, 8 | 16 | 32) {
        return Err(format!("ZBITPIX {} is not handled on the GPU", t.zbitpix));
    }
    let ntx = t.naxis1.div_ceil(t.tile1);
    let n = t.nrows;
    let mut job = RiceJob {
        heap_file_offset: t.heap_offset as u64,
        heap_len: 0,
        offsets: Vec::with_capacity(n),
        lens: Vec::with_capacity(n),
        npix: Vec::with_capacity(n),
        scale: Vec::with_capacity(n),
        zero: Vec::with_capacity(n),
        tnull: Vec::with_capacity(n),
        blocksize: t.blocksize as i32,
        bytepix: t.bytepix as i32,
        dither: t.dither as i32,
        dither_seed: t.dither_seed,
        is_float,
        f32_out: t.zbitpix == -32,
        bzero: t.bzero,
        bscale: t.bscale,
    };
    let mut heap_end = 0usize;
    for row in 0..n {
        let (count, off) = heap_descriptor(t, mmap, &comp, row).map_err(|e| e.to_string())?;
        if count == 0 {
            return Err("some tiles are stored without Rice compression: not handled on the GPU yet".to_string());
        }
        heap_end = heap_end.max(off + count);
        job.offsets.push(off as u64);
        job.lens.push(count as u32);
        let (tx, ty) = (row % ntx, row / ntx);
        let w = t.tile1.min(t.naxis1 - tx * t.tile1);
        let h = t.tile2.min(t.naxis2 - ty * t.tile2);
        job.npix.push((w * h) as u32);
        if is_float {
            job.scale.push(column_f64(t, mmap, t.zscale.as_ref().unwrap(), row).map_err(|e| e.to_string())?);
            job.zero.push(column_f64(t, mmap, t.zzero.as_ref().unwrap(), row).map_err(|e| e.to_string())?);
        } else {
            job.scale.push(0.0);
            job.zero.push(0.0);
        }
        let null = t.zblank_col.as_ref().and_then(|c| column_i64(t, mmap, c, row)).or(t.zblank);
        job.tnull.push(null.unwrap_or(i64::MIN)); // no pixel can equal i64::MIN
    }
    job.heap_len = heap_end;
    Ok(job)
}

/// Why this image cannot be handled by the GPU code, if it cannot.
fn unsupported_reason(header: &FitsHeader) -> Option<String> {
    if header.tiled.is_some() {
        return None; // Rice images are checked in `rice_job`
    }
    if !matches!(header.bitpix, 8 | 16 | 32 | -32 | -64) {
        return Some(format!("BITPIX {} is not handled on the GPU yet (use the CPU for 64-bit integers)", header.bitpix));
    }
    None
}

fn to_stats(header: &FitsHeader, raw: Raw, device: String, total_ms: f64) -> FitsStats {
    let n_pixels = header.naxis1 * header.naxis2;
    let nans = raw.nans as usize;
    let valid = n_pixels - nans.min(n_pixels);
    let (min, max, mean) = if valid == 0 {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        (raw.min, raw.max, raw.sum / valid as f64)
    };
    FitsStats {
        naxis1: header.naxis1,
        naxis2: header.naxis2,
        bitpix: header.bitpix,
        min,
        max,
        mean,
        min_int: None,
        max_int: None,
        n_nan: nans,
        threads_used: 0,
        elapsed_ms: total_ms,
        device,
        upload_ms: raw.upload_ms,
        kernel_ms: raw.kernel_ms,
    }
}

/// Statistics on the GPU; an error says why they could not be computed there.
fn stats_gpu(path: &str, hdu: &Option<HduSelector>) -> Result<FitsStats, String> {
    let start = Instant::now();
    let mmap = open_mmap(path).map_err(|e| e.to_string())?;
    let header = locate_image(&mmap, hdu).map_err(|e| e.to_string())?;
    if let Some(why) = unsupported_reason(&header) {
        return Err(why);
    }
    if header.tiled.is_some() {
        let job = rice_job(&mmap, &header)?;
        let raw = backend::stats_rice(path, &job)?;
        return Ok(to_stats(&header, raw, backend::device_name(), start.elapsed().as_secs_f64() * 1000.0));
    }
    let bytes = (header.bitpix.unsigned_abs() as usize) / 8;
    let end = header.data_offset + header.naxis1 * header.naxis2 * bytes;
    if end > mmap.len() {
        return Err("truncated file: fewer pixels than the header announces".to_string());
    }
    let debug = std::env::var_os("EUCLIDRUST_GPU_DEBUG").is_some();
    let t_opened = start.elapsed().as_secs_f64() * 1000.0;
    let raw = backend::stats_chunked(path, header.data_offset as u64, end - header.data_offset, header.bitpix, header.bzero, header.bscale)?;
    let t_done = start.elapsed().as_secs_f64() * 1000.0;
    let stats = to_stats(&header, raw, backend::device_name(), t_done);
    let t0 = Instant::now();
    drop(mmap);
    if debug {
        eprintln!(
            "[gpu] file opened and header read after {t_opened:.1} ms | statistics done at {t_done:.1} ms | closing the file mapping took {:.1} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(stats)
}

/// Whether `device="auto"` should send this image to the GPU. CUDA is only started (and so
/// only costs anything) once the image itself looks like one the GPU is good at.
fn worth_the_gpu(path: &str, hdu: &Option<HduSelector>) -> bool {
    let Ok(mmap) = open_mmap(path) else { return false };
    let Ok(header) = locate_image(&mmap, hdu) else { return false };
    let Some(t) = &header.tiled else { return false };
    if t.algo != Algo::Rice || t.nrows < AUTO_MIN_TILES || mmap.len() < AUTO_MIN_FILE_BYTES {
        return false;
    }
    backend::probe().available
}

/// The entry point used by `fits_stats(device=...)`.
pub fn stats_with_device(path: &str, hdu: &Option<HduSelector>, device: Device) -> PyResult<FitsStats> {
    match device {
        Device::Cpu => stats_impl(path, hdu),
        Device::Gpu => stats_gpu(path, hdu).map_err(|e| PyValueError::new_err(format!("GPU: {e}"))),
        Device::Auto => {
            if worth_the_gpu(path, hdu) {
                if let Ok(s) = stats_gpu(path, hdu) {
                    return Ok(s);
                }
            }
            stats_impl(path, hdu)
        }
    }
}

/// An image kept in the GPU's memory, for repeated computations.
#[pyclass]
pub struct GpuImage {
    res: backend::Resident,
    header: FitsHeader,
    #[pyo3(get)]
    upload_ms: f64,
    #[pyo3(get)]
    nbytes: usize,
}

#[pymethods]
impl GpuImage {
    /// Copies the image (the primary HDU by default) to the GPU.
    #[new]
    #[pyo3(signature = (path, hdu=None))]
    fn new(python: Python<'_>, path: &str, hdu: Option<HduSelector>) -> PyResult<Self> {
        python
            .detach(|| -> Result<GpuImage, String> {
                let mmap = open_mmap(path).map_err(|e| e.to_string())?;
                let header = locate_image(&mmap, &hdu).map_err(|e| e.to_string())?;
                if header.tiled.is_some() {
                    return Err("a tile-compressed image cannot be kept on the GPU: it is decoded tile by tile as it is used".to_string());
                }
                if let Some(why) = unsupported_reason(&header) {
                    return Err(why);
                }
                let bytes = (header.bitpix.unsigned_abs() as usize) / 8;
                let end = header.data_offset + header.naxis1 * header.naxis2 * bytes;
                if end > mmap.len() {
                    return Err("truncated file: fewer pixels than the header announces".to_string());
                }
                let res = backend::upload(path, header.data_offset as u64, end - header.data_offset)?;
                Ok(GpuImage { upload_ms: res.upload_ms, nbytes: res.len, res, header })
            })
            .map_err(|e| PyValueError::new_err(format!("GPU: {e}")))
    }

    /// Statistics computed on the card, with no copy: only the kernel runs.
    fn stats(&self, python: Python<'_>) -> PyResult<FitsStats> {
        python
            .detach(|| {
                let start = Instant::now();
                let raw = backend::stats_resident(&self.res, self.header.bitpix, self.header.bzero, self.header.bscale)?;
                Ok(to_stats(&self.header, raw, backend::device_name(), start.elapsed().as_secs_f64() * 1000.0))
            })
            .map_err(|e: String| PyValueError::new_err(format!("GPU: {e}")))
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.header.naxis1, self.header.naxis2)
    }
}
