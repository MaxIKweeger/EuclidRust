use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;

const CARD_LEN: usize = 80;
const BLOCK_LEN: usize = 2880; // 36 cards of 80 bytes
const MAX_HEADER_BLOCKS: usize = 200; // safety guard: a sane header is never this long
const MAX_HDUS: usize = 100_000;
const N_PV: usize = 40; // PVi_0 .. PVi_39, the TPV polynomial terms

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

/// Which HDU (header/data unit) to read: a 0-based index (`hdu=1`) or an
/// `EXTNAME` (`hdu="DET11.SCI"`). Python passes an `int` or a `str`.
#[derive(FromPyObject)]
enum HduSelector {
    Index(usize),
    Name(String),
}

/// TPV distortion: the polynomial coefficients PV1_m / PV2_m applied to the
/// intermediate world coordinates (the "TPV" convention used by SCAMP/SWarp
/// and by the Euclid VIS pipeline).
#[derive(Debug, Clone, Copy)]
struct Tpv {
    pv1: [f64; N_PV],
    pv2: [f64; N_PV],
}

/// WCS (World Coordinate System) parameters: the standard FITS
/// "pixel <-> sky coordinates" transformation, using the tangent (TAN)
/// projection, optionally with TPV polynomial distortion. Absent (`None`) if
/// the header has no usable astrometry keywords.
#[derive(Debug, Clone, Copy)]
struct Wcs {
    crval1: f64, // RA of the reference pixel, in degrees
    crval2: f64, // Dec of the reference pixel, in degrees
    crpix1: f64, // reference pixel, axis 1 (FITS convention, 1-indexed)
    crpix2: f64, // reference pixel, axis 2
    cd1_1: f64,  // WCS matrix, degrees/pixel
    cd1_2: f64,
    cd2_1: f64,
    cd2_2: f64,
    tpv: Option<Tpv>, // Some(..) when CTYPE is RA---TPV / DEC--TPV
}

/// Everything we read from one HDU header.
#[derive(Debug, Clone)]
struct RawHeader {
    xtension: Option<String>,
    extname: Option<String>,
    zimage: bool,
    bitpix: Option<i32>,
    naxis: Option<usize>,
    naxes: Vec<usize>, // NAXIS1, NAXIS2, ...
    pcount: usize,
    gcount: usize,
    bzero: f64,
    bscale: f64,
    crval1: Option<f64>,
    crval2: Option<f64>,
    crpix1: Option<f64>,
    crpix2: Option<f64>,
    cd1_1: Option<f64>,
    cd1_2: Option<f64>,
    cd2_1: Option<f64>,
    cd2_2: Option<f64>,
    ctype1: Option<String>,
    ctype2: Option<String>,
    pv1: Vec<Option<f64>>,
    pv2: Vec<Option<f64>>,
}

impl RawHeader {
    fn new() -> Self {
        RawHeader {
            xtension: None,
            extname: None,
            zimage: false,
            bitpix: None,
            naxis: None,
            naxes: Vec::new(),
            pcount: 0,
            gcount: 1,
            bzero: 0.0,
            bscale: 1.0,
            crval1: None,
            crval2: None,
            crpix1: None,
            crpix2: None,
            cd1_1: None,
            cd1_2: None,
            cd2_1: None,
            cd2_2: None,
            ctype1: None,
            ctype2: None,
            pv1: vec![None; N_PV],
            pv2: vec![None; N_PV],
        }
    }
}

/// A located HDU: its parsed header and the byte offset where its data starts.
struct Hdu {
    index: usize,
    header: RawHeader,
    data_offset: usize,
}

/// The pixel array of the selected HDU plus what is needed to read it.
#[derive(Debug, Clone)]
struct FitsHeader {
    bitpix: i32,
    naxis1: usize,
    naxis2: usize,
    bzero: f64,
    bscale: f64,
    data_offset: usize, // byte offset where the pixels start
    wcs: Option<Wcs>,
    ctype1: Option<String>, // projection of axis 1, e.g. "RA---TAN"
    ctype2: Option<String>, // projection of axis 2, e.g. "DEC--TAN"
}

fn parse_f64(s: &str) -> Option<f64> {
    // FITS allows a Fortran-style "D" exponent (1.0D+00)
    s.replace(['D', 'd'], "E").parse::<f64>().ok()
}

fn unquote(s: &str) -> String {
    s.trim_matches('\'').trim().to_string()
}

/// Parses the header cards of the HDU starting at `offset` (2880-byte blocks
/// = 36 cards of 80 bytes) up to its "END" card. Returns the parsed header
/// and the byte offset where the HDU's data starts.
fn read_header(mmap: &[u8], offset: usize) -> PyResult<(RawHeader, usize)> {
    let mut h = RawHeader::new();
    let mut block_start = offset;
    for _ in 0..MAX_HEADER_BLOCKS {
        if block_start + BLOCK_LEN > mmap.len() {
            return err("truncated file: no END card found");
        }
        let block = &mmap[block_start..block_start + BLOCK_LEN];
        for card_idx in 0..(BLOCK_LEN / CARD_LEN) {
            let card = &block[card_idx * CARD_LEN..(card_idx + 1) * CARD_LEN];
            let card_str = String::from_utf8_lossy(card);
            let keyword = card_str.get(0..8).unwrap_or("").trim();

            if keyword == "END" {
                return Ok((h, block_start + BLOCK_LEN));
            }

            if card_str.get(8..10) == Some("= ") {
                let value_part = card_str.get(10..).unwrap_or("");
                let value_str = value_part.split('/').next().unwrap_or("").trim();
                match keyword {
                    "XTENSION" => h.xtension = Some(unquote(value_str)),
                    "EXTNAME" => h.extname = Some(unquote(value_str)),
                    "ZIMAGE" => h.zimage = value_str == "T",
                    "BITPIX" => h.bitpix = value_str.parse::<i32>().ok(),
                    "NAXIS" => h.naxis = value_str.parse::<usize>().ok(),
                    "PCOUNT" => h.pcount = value_str.parse::<usize>().unwrap_or(0),
                    "GCOUNT" => h.gcount = value_str.parse::<usize>().unwrap_or(1),
                    "BZERO" => h.bzero = parse_f64(value_str).unwrap_or(0.0),
                    "BSCALE" => h.bscale = parse_f64(value_str).unwrap_or(1.0),
                    "CRVAL1" => h.crval1 = parse_f64(value_str),
                    "CRVAL2" => h.crval2 = parse_f64(value_str),
                    "CRPIX1" => h.crpix1 = parse_f64(value_str),
                    "CRPIX2" => h.crpix2 = parse_f64(value_str),
                    "CD1_1" => h.cd1_1 = parse_f64(value_str),
                    "CD1_2" => h.cd1_2 = parse_f64(value_str),
                    "CD2_1" => h.cd2_1 = parse_f64(value_str),
                    "CD2_2" => h.cd2_2 = parse_f64(value_str),
                    "CTYPE1" => h.ctype1 = Some(unquote(value_str)),
                    "CTYPE2" => h.ctype2 = Some(unquote(value_str)),
                    _ => {
                        if let Some(rest) = keyword.strip_prefix("NAXIS") {
                            if let Ok(i) = rest.parse::<usize>() {
                                if (1..=999).contains(&i) {
                                    if h.naxes.len() < i {
                                        h.naxes.resize(i, 0);
                                    }
                                    h.naxes[i - 1] = value_str.parse::<usize>().unwrap_or(0);
                                }
                            }
                        } else if let Some(rest) = keyword.strip_prefix("PV1_") {
                            if let Ok(i) = rest.parse::<usize>() {
                                if i < N_PV {
                                    h.pv1[i] = parse_f64(value_str);
                                }
                            }
                        } else if let Some(rest) = keyword.strip_prefix("PV2_") {
                            if let Ok(i) = rest.parse::<usize>() {
                                if i < N_PV {
                                    h.pv2[i] = parse_f64(value_str);
                                }
                            }
                        }
                    }
                }
            }
        }
        block_start += BLOCK_LEN;
    }
    err("END card not found")
}

/// Size in bytes of an HDU's data, padded to a whole number of 2880-byte blocks.
fn padded_data_len(h: &RawHeader) -> usize {
    let naxis = h.naxis.unwrap_or(0);
    if naxis == 0 {
        return 0;
    }
    let bytes_per_value = (h.bitpix.unwrap_or(8).unsigned_abs() as usize) / 8;
    let n: usize = h.naxes.iter().take(naxis).fold(1usize, |a, &b| a.saturating_mul(b));
    let raw = bytes_per_value
        .saturating_mul(h.gcount.max(1))
        .saturating_mul(h.pcount.saturating_add(n));
    raw.div_ceil(BLOCK_LEN).saturating_mul(BLOCK_LEN)
}

/// Walks every HDU of the file (reading headers only, never the data).
fn all_hdus(mmap: &[u8]) -> PyResult<Vec<Hdu>> {
    let mut hdus = Vec::new();
    let mut offset = 0usize;
    while offset + BLOCK_LEN <= mmap.len() && hdus.len() < MAX_HDUS {
        match read_header(mmap, offset) {
            Ok((header, data_offset)) => {
                let next = data_offset.saturating_add(padded_data_len(&header));
                hdus.push(Hdu { index: hdus.len(), header, data_offset });
                offset = next;
            }
            Err(e) => {
                if hdus.is_empty() {
                    return Err(e);
                }
                break; // trailing garbage / truncated last HDU: keep what we have
            }
        }
    }
    Ok(hdus)
}

fn is_compressed_image(h: &RawHeader) -> bool {
    h.xtension.as_deref() == Some("BINTABLE") && h.zimage
}

fn is_image_hdu(h: &RawHeader) -> bool {
    matches!(h.xtension.as_deref(), None | Some("IMAGE"))
}

fn hdu_kind(h: &RawHeader) -> String {
    h.xtension.clone().unwrap_or_else(|| "PRIMARY".to_string())
}

fn shape_str(h: &RawHeader) -> String {
    let n = h.naxis.unwrap_or(0);
    h.naxes.iter().take(n).map(|d| d.to_string()).collect::<Vec<_>>().join("x")
}

fn describe(hdu: &Hdu) -> String {
    let h = &hdu.header;
    let name = h.extname.clone().unwrap_or_default();
    let label = if name.is_empty() { String::new() } else { format!(" {name}") };
    if is_compressed_image(h) {
        format!("[{}]{} (tile-compressed image, not supported yet)", hdu.index, label)
    } else {
        format!("[{}]{} ({}, BITPIX {})", hdu.index, label, shape_str(h), h.bitpix.unwrap_or(0))
    }
}

/// A short list of the image HDUs of a file, for error messages.
fn image_hdu_summary(hdus: &[Hdu]) -> String {
    let imgs: Vec<String> = hdus
        .iter()
        .filter(|h| is_compressed_image(&h.header) || (is_image_hdu(&h.header) && h.header.naxis.unwrap_or(0) >= 1))
        .map(describe)
        .collect();
    if imgs.is_empty() {
        return "it contains no image HDU".to_string();
    }
    let shown: Vec<&str> = imgs.iter().take(12).map(|s| s.as_str()).collect();
    let more = if imgs.len() > 12 { format!(" ... (+{} more)", imgs.len() - 12) } else { String::new() };
    format!("it contains: {}{}", shown.join("; "), more)
}

fn build_wcs(h: &RawHeader) -> Option<Wcs> {
    match (h.crval1, h.crval2, h.crpix1, h.crpix2, h.cd1_1, h.cd2_2) {
        (Some(crval1), Some(crval2), Some(crpix1), Some(crpix2), Some(cd1_1), Some(cd2_2)) => {
            let is_tpv = h.ctype1.as_deref() == Some("RA---TPV") && h.ctype2.as_deref() == Some("DEC--TPV");
            let tpv = if is_tpv {
                let coeffs = |pv: &Vec<Option<f64>>| {
                    let mut a = [0.0f64; N_PV];
                    a[1] = 1.0; // PVi_1 defaults to 1 (identity) when absent
                    for (k, v) in pv.iter().enumerate() {
                        if let Some(v) = v {
                            a[k] = *v;
                        }
                    }
                    a
                };
                Some(Tpv { pv1: coeffs(&h.pv1), pv2: coeffs(&h.pv2) })
            } else {
                None
            };
            Some(Wcs {
                crval1,
                crval2,
                crpix1,
                crpix2,
                cd1_1,
                cd1_2: h.cd1_2.unwrap_or(0.0),
                cd2_1: h.cd2_1.unwrap_or(0.0),
                cd2_2,
                tpv,
            })
        }
        _ => None,
    }
}

/// Finds the requested HDU, checks that it is a plain uncompressed 2-D image
/// and returns what is needed to read its pixels. With `hdu=None` the
/// primary HDU is used.
fn locate_image(mmap: &[u8], sel: &Option<HduSelector>) -> PyResult<FitsHeader> {
    let implicit_primary = matches!(sel, None | Some(HduSelector::Index(0)));

    let (hdu, label): (Hdu, String) = if implicit_primary {
        let (header, data_offset) = read_header(mmap, 0)?;
        (Hdu { index: 0, header, data_offset }, "the primary HDU".to_string())
    } else {
        let hdus = all_hdus(mmap)?;
        let n = hdus.len();
        let found = match sel {
            Some(HduSelector::Index(i)) => {
                if *i >= n {
                    return err(format!("no HDU with index {i}: the file has {n} HDU(s) (0..{})", n - 1));
                }
                *i
            }
            Some(HduSelector::Name(name)) => {
                match hdus.iter().position(|x| x.header.extname.as_deref().is_some_and(|e| e.eq_ignore_ascii_case(name))) {
                    Some(p) => p,
                    None => return err(format!("no HDU named '{name}': {}", image_hdu_summary(&hdus))),
                }
            }
            _ => unreachable!(),
        };
        let hdu = hdus.into_iter().nth(found).unwrap();
        let label = match &hdu.header.extname {
            Some(name) => format!("HDU {} ({name})", hdu.index),
            None => format!("HDU {}", hdu.index),
        };
        (hdu, label)
    };
    let h = &hdu.header;

    if is_compressed_image(h) {
        return err(format!("{label} is a tile-compressed image (ZIMAGE), which is not supported yet"));
    }
    if !is_image_hdu(h) {
        return err(format!("{label} is a {} table, not an image", hdu_kind(h)));
    }
    let naxis = h.naxis.ok_or_else(|| PyValueError::new_err("missing NAXIS"))?;
    if naxis == 0 {
        if implicit_primary {
            let summary = match all_hdus(mmap) {
                Ok(hdus) => image_hdu_summary(&hdus),
                Err(_) => "it contains no image HDU".to_string(),
            };
            return err(format!(
                "the primary HDU holds no image (NAXIS = 0); {summary}. \
                 Pass hdu=<index or EXTNAME> to choose one, e.g. hdu=1"
            ));
        }
        return err(format!("{label} holds no image (NAXIS = 0)"));
    }
    if naxis != 2 {
        return err(format!(
            "NAXIS = {naxis} is not supported ({label}): only 2-D images are handled (3-D cubes are not)"
        ));
    }
    let bitpix = h.bitpix.ok_or_else(|| PyValueError::new_err("missing BITPIX"))?;
    let naxis1 = h.naxes.first().copied().filter(|&v| v > 0).ok_or_else(|| PyValueError::new_err("missing NAXIS1"))?;
    let naxis2 = h.naxes.get(1).copied().filter(|&v| v > 0).ok_or_else(|| PyValueError::new_err("missing NAXIS2"))?;

    Ok(FitsHeader {
        bitpix,
        naxis1,
        naxis2,
        bzero: h.bzero,
        bscale: h.bscale,
        data_offset: hdu.data_offset,
        wcs: build_wcs(h),
        ctype1: h.ctype1.clone(),
        ctype2: h.ctype2.clone(),
    })
}

fn open_mmap(path: &str) -> PyResult<Mmap> {
    let file = File::open(path).map_err(|e| PyValueError::new_err(format!("cannot open {path}: {e}")))?;
    unsafe { Mmap::map(&file).map_err(|e| PyValueError::new_err(format!("mmap failed: {e}"))) }
}

/// One HDU of a FITS file, as reported by `list_hdus`.
#[pyclass]
struct HduInfo {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    kind: String, // "PRIMARY", "IMAGE", "BINTABLE", "TABLE"...
    #[pyo3(get)]
    name: String, // EXTNAME, or "" if none
    #[pyo3(get)]
    bitpix: Option<i32>,
    #[pyo3(get)]
    shape: Vec<usize>, // (NAXIS1, NAXIS2, ...) in FITS order
    #[pyo3(get)]
    readable: bool, // True if euclid_fast can read it (uncompressed 2-D image, supported BITPIX)
    #[pyo3(get)]
    note: String, // why it is not readable, or ""
}

#[pymethods]
impl HduInfo {
    fn __repr__(&self) -> String {
        let name = if self.name.is_empty() { String::new() } else { format!(" name={:?}", self.name) };
        let shape = self.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
        let note = if self.note.is_empty() { String::new() } else { format!(" ({})", self.note) };
        format!(
            "<HduInfo index={}{} kind={} shape={} bitpix={} readable={}{}>",
            self.index,
            name,
            self.kind,
            if shape.is_empty() { "-".to_string() } else { shape },
            self.bitpix.map_or("-".to_string(), |b| b.to_string()),
            if self.readable { "yes" } else { "no" },
            note
        )
    }
}

/// Lists the HDUs of a FITS file (reading headers only), so you can see which
/// ones `euclid_fast` can read and what to pass as `hdu=`.
#[pyfunction]
fn list_hdus(path: &str) -> PyResult<Vec<HduInfo>> {
    let mmap = open_mmap(path)?;
    let hdus = all_hdus(&mmap)?;
    Ok(hdus
        .iter()
        .map(|x| {
            let h = &x.header;
            let naxis = h.naxis.unwrap_or(0);
            let supported_bitpix = matches!(h.bitpix, Some(8) | Some(16) | Some(32) | Some(64) | Some(-32) | Some(-64));
            let (readable, note) = if is_compressed_image(h) {
                (false, "tile-compressed image, not supported yet".to_string())
            } else if !is_image_hdu(h) {
                (false, "table, not an image".to_string())
            } else if naxis == 0 {
                (false, "no image data".to_string())
            } else if naxis != 2 {
                (false, format!("NAXIS = {naxis}, only 2-D images are supported"))
            } else if !supported_bitpix {
                (false, format!("BITPIX {} is not supported", h.bitpix.unwrap_or(0)))
            } else {
                (true, String::new())
            };
            HduInfo {
                index: x.index,
                kind: hdu_kind(h),
                name: h.extname.clone().unwrap_or_default(),
                bitpix: h.bitpix,
                shape: h.naxes.iter().take(naxis).copied().collect(),
                readable,
                note,
            }
        })
        .collect())
}

/// Image statistics: min, max, mean (physical values, after applying
/// BZERO/BSCALE), plus some context about the computation.
#[pyclass]
struct FitsStats {
    #[pyo3(get)]
    naxis1: usize,
    #[pyo3(get)]
    naxis2: usize,
    #[pyo3(get)]
    bitpix: i32,
    #[pyo3(get)]
    min: f64,
    #[pyo3(get)]
    max: f64,
    #[pyo3(get)]
    mean: f64,
    #[pyo3(get)]
    threads_used: usize,
    #[pyo3(get)]
    elapsed_ms: f64,
}

/// Reads a FITS image (uncompressed, 2-D) through mmap (zero-copy for the
/// disk -> memory part) and computes min/max/mean in parallel with rayon.
/// The rayon thread pool sizes itself to the number of cores available on
/// the machine (no hardcoded value). `hdu` selects the HDU by 0-based index
/// or EXTNAME; by default the primary HDU is read.
#[pyfunction]
#[pyo3(signature = (path, hdu=None))]
fn fits_stats(path: &str, hdu: Option<HduSelector>) -> PyResult<FitsStats> {
    let start = std::time::Instant::now();

    let mmap = open_mmap(path)?;
    let header = locate_image(&mmap, &hdu)?;
    let n_pixels = header.naxis1 * header.naxis2;
    let bytes_per_pixel = (header.bitpix.unsigned_abs() as usize) / 8;
    let data_end = header.data_offset + n_pixels * bytes_per_pixel;

    if data_end > mmap.len() {
        return err("truncated file: fewer pixels than the header announces");
    }
    let data = &mmap[header.data_offset..data_end];

    // Read + convert to "physical" f64 (BZERO + BSCALE * raw_value), in
    // parallel over chunks of pixels. rayon sizes the number of threads
    // itself from the machine's logical core count.
    let (min, max, sum) = match header.bitpix {
        16 => reduce_be::<i16>(data, header.bzero, header.bscale),
        8 => reduce_be::<u8>(data, header.bzero, header.bscale),
        32 => reduce_be::<i32>(data, header.bzero, header.bscale),
        64 => reduce_be::<i64>(data, header.bzero, header.bscale),
        -32 => reduce_be::<f32>(data, header.bzero, header.bscale),
        -64 => reduce_be::<f64>(data, header.bzero, header.bscale),
        other => return err(format!("BITPIX {other} is not supported yet")),
    };

    let mean = sum / n_pixels as f64;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(FitsStats {
        naxis1: header.naxis1,
        naxis2: header.naxis2,
        bitpix: header.bitpix,
        min,
        max,
        mean,
        threads_used: rayon::current_num_threads(),
        elapsed_ms,
    })
}

/// Minimal trait to convert a raw big-endian sample into f64.
trait BigEndianSample: Copy {
    const SIZE: usize;
    fn from_be_slice(b: &[u8]) -> f64;
}

impl BigEndianSample for i16 {
    const SIZE: usize = 2;
    fn from_be_slice(b: &[u8]) -> f64 {
        i16::from_be_bytes([b[0], b[1]]) as f64
    }
}
impl BigEndianSample for u8 {
    const SIZE: usize = 1;
    fn from_be_slice(b: &[u8]) -> f64 {
        b[0] as f64
    }
}
impl BigEndianSample for i32 {
    const SIZE: usize = 4;
    fn from_be_slice(b: &[u8]) -> f64 {
        i32::from_be_bytes([b[0], b[1], b[2], b[3]]) as f64
    }
}
// 64-bit integers are converted to f64, which is exact up to 2^53.
impl BigEndianSample for i64 {
    const SIZE: usize = 8;
    fn from_be_slice(b: &[u8]) -> f64 {
        i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64
    }
}
impl BigEndianSample for f32 {
    const SIZE: usize = 4;
    fn from_be_slice(b: &[u8]) -> f64 {
        f32::from_be_bytes([b[0], b[1], b[2], b[3]]) as f64
    }
}
impl BigEndianSample for f64 {
    const SIZE: usize = 8;
    fn from_be_slice(b: &[u8]) -> f64 {
        f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
}

/// Parallel reduction (min, max, sum) over the raw pixels of type T,
/// applying BZERO/BSCALE to get the physical value.
///
/// v2: instead of handing rayon a per-pixel iterator (per-element
/// scheduling/fold overhead), the data is split into large blocks
/// (`par_chunks`), each processed by a tight sequential loop -- the loop
/// style LLVM auto-vectorizes best (implicit SIMD, no explicit intrinsics,
/// no nightly) -- while keeping parallelism across blocks via rayon.
fn reduce_be<T: BigEndianSample + Sync>(data: &[u8], bzero: f64, bscale: f64) -> (f64, f64, f64) {
    const CHUNK_PIXELS: usize = 1 << 16; // 65536 pixels per rayon block
    let chunk_bytes = CHUNK_PIXELS * T::SIZE;

    data.par_chunks(chunk_bytes.max(T::SIZE))
        .map(|chunk| {
            let n = chunk.len() / T::SIZE;
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            let mut sum = 0.0_f64;
            for i in 0..n {
                let off = i * T::SIZE;
                let v = bzero + bscale * T::from_be_slice(&chunk[off..off + T::SIZE]);
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
                sum += v;
            }
            (mn, mx, sum)
        })
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64),
            |(mn1, mx1, s1), (mn2, mx2, s2)| (mn1.min(mn2), mx1.max(mx2), s1 + s2),
        )
}

/// The (powers of x, powers of y, powers of r) of each of the 40 TPV terms,
/// in the order of the PVi_0 .. PVi_39 keywords.
const TPV_TERMS: [(i32, i32, i32); N_PV] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (1, 1, 0),
    (0, 2, 0),
    (3, 0, 0),
    (2, 1, 0),
    (1, 2, 0),
    (0, 3, 0),
    (0, 0, 3),
    (4, 0, 0),
    (3, 1, 0),
    (2, 2, 0),
    (1, 3, 0),
    (0, 4, 0),
    (5, 0, 0),
    (4, 1, 0),
    (3, 2, 0),
    (2, 3, 0),
    (1, 4, 0),
    (0, 5, 0),
    (0, 0, 5),
    (6, 0, 0),
    (5, 1, 0),
    (4, 2, 0),
    (3, 3, 0),
    (2, 4, 0),
    (1, 5, 0),
    (0, 6, 0),
    (7, 0, 0),
    (6, 1, 0),
    (5, 2, 0),
    (4, 3, 0),
    (3, 4, 0),
    (2, 5, 0),
    (1, 6, 0),
    (0, 7, 0),
    (0, 0, 7),
];

/// Evaluates one TPV polynomial: sum of pv[k] * x^a * y^b * r^c.
fn tpv_poly(pv: &[f64; N_PV], x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    let mut sum = 0.0;
    for (k, &(a, b, c)) in TPV_TERMS.iter().enumerate() {
        if pv[k] != 0.0 {
            sum += pv[k] * x.powi(a) * y.powi(b) * r.powi(c);
        }
    }
    sum
}

/// TPV forward distortion: intermediate world coordinates (x, y), in
/// degrees, -> distorted ones (xi', eta'). Axis 2 swaps the roles of x and y.
fn tpv_forward(t: &Tpv, x: f64, y: f64) -> (f64, f64) {
    (tpv_poly(&t.pv1, x, y), tpv_poly(&t.pv2, y, x))
}

/// Inverts the TPV distortion with Newton's method: finds (x, y) such that
/// tpv_forward(x, y) = (xi, eta). Converges to ~1e-12 degree (4e-9 arcsec).
fn tpv_invert(t: &Tpv, xi: f64, eta: f64) -> PyResult<(f64, f64)> {
    let (mut x, mut y) = (xi, eta);
    let h = 1e-6; // finite-difference step, degrees
    for _ in 0..50 {
        let (fx, fy) = tpv_forward(t, x, y);
        let (rx, ry) = (fx - xi, fy - eta);
        if rx.abs() < 1e-12 && ry.abs() < 1e-12 {
            return Ok((x, y));
        }
        let (fx_dx, fy_dx) = tpv_forward(t, x + h, y);
        let (fx_dy, fy_dy) = tpv_forward(t, x, y + h);
        let (a, c) = ((fx_dx - fx) / h, (fy_dx - fy) / h); // d(fx,fy)/dx
        let (b, d) = ((fx_dy - fx) / h, (fy_dy - fy) / h); // d(fx,fy)/dy
        let det = a * d - b * c;
        if det == 0.0 || !det.is_finite() {
            return err("TPV distortion cannot be inverted at this position (singular Jacobian)");
        }
        x -= (d * rx - b * ry) / det;
        y -= (-c * rx + a * ry) / det;
    }
    err("TPV distortion inversion did not converge")
}

/// Converts sky coordinates (RA/Dec, in degrees) to pixel coordinates (FITS
/// 1-indexed convention) using the tangent (TAN / gnomonic) projection --
/// the projection used by almost all modern surveys (Euclid included) --
/// plus the TPV polynomial distortion when the header has one.
/// SIP distortion is not handled.
fn radec_to_pixel(ra_deg: f64, dec_deg: f64, wcs: &Wcs) -> PyResult<(f64, f64)> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let ra0 = wcs.crval1.to_radians();
    let dec0 = wcs.crval2.to_radians();

    let cos_c = dec0.sin() * dec.sin() + dec0.cos() * dec.cos() * (ra - ra0).cos();
    if cos_c <= 1e-9 {
        return err("the requested sky position is not within 90 degrees of the image center (no tangent-plane projection)");
    }
    let xi = dec.cos() * (ra - ra0).sin() / cos_c; // radians
    let eta = (dec0.cos() * dec.sin() - dec0.sin() * dec.cos() * (ra - ra0).cos()) / cos_c;

    let xi_deg = xi.to_degrees();
    let eta_deg = eta.to_degrees();

    // With TPV, the tangent-plane coordinates above are the *distorted* ones:
    // undo the polynomial to get the intermediate world coordinates.
    let (xw, yw) = match &wcs.tpv {
        Some(t) => tpv_invert(t, xi_deg, eta_deg)?,
        None => (xi_deg, eta_deg),
    };

    // Invert the 2x2 CD matrix to go from intermediate world coordinates
    // to pixel offsets.
    let det = wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1;
    let dx = (wcs.cd2_2 * xw - wcs.cd1_2 * yw) / det;
    let dy = (-wcs.cd2_1 * xw + wcs.cd1_1 * yw) / det;

    Ok((wcs.crpix1 + dx, wcs.crpix2 + dy))
}

/// Result of a cutout extraction: the sub-image (physical values, row by
/// row) plus context (where it was taken, how long it took).
#[pyclass]
struct Cutout {
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
    #[pyo3(get)]
    pixel_x_center: f64,
    #[pyo3(get)]
    pixel_y_center: f64,
    #[pyo3(get)]
    data: Vec<f64>, // row-major, length = width * height
    #[pyo3(get)]
    elapsed_ms: f64,
}

/// Shared core of the cutout extraction (without timing or the PyO3
/// wrapping), reused by `extract_cutout` and `export_cutout_parquet`.
/// Returns (pixel_x_center, pixel_y_center, x0, y0, row-major data).
fn extract_cutout_raw(
    path: &str,
    ra_deg: f64,
    dec_deg: f64,
    width: usize,
    height: usize,
    hdu: &Option<HduSelector>,
) -> PyResult<(f64, f64, usize, usize, Vec<f64>)> {
    let mmap = open_mmap(path)?;
    let header = locate_image(&mmap, hdu)?;
    let wcs = header
        .wcs
        .ok_or_else(|| PyValueError::new_err("this file has no WCS header (CRVAL/CRPIX/CD)"))?;

    // Only the tangent projection (plain TAN, or TPV with its polynomial) is
    // implemented. Anything else (SIN, ARC, TAN-SIP...) would silently give
    // wrong positions.
    let ct = (header.ctype1.as_deref(), header.ctype2.as_deref());
    if ct != (Some("RA---TAN"), Some("DEC--TAN")) && ct != (Some("RA---TPV"), Some("DEC--TPV")) {
        return err(format!(
            "unsupported WCS projection: expected CTYPE1/CTYPE2 = RA---TAN/DEC--TAN or RA---TPV/DEC--TPV, found {}/{} \
             (other projections and SIP distortion are not supported)",
            ct.0.unwrap_or("<missing>"),
            ct.1.unwrap_or("<missing>"),
        ));
    }

    let (px, py) = radec_to_pixel(ra_deg, dec_deg, &wcs)?;
    // Convert FITS 1-indexed -> 0-indexed array, then take the cutout's top-left corner.
    let cx = px - 1.0;
    let cy = py - 1.0;
    let x0 = (cx - width as f64 / 2.0).round() as i64;
    let y0 = (cy - height as f64 / 2.0).round() as i64;

    if x0 < 0
        || y0 < 0
        || x0 as usize + width > header.naxis1
        || y0 as usize + height > header.naxis2
    {
        return err(format!(
            "cutout outside the image: center pixel=({px:.1},{py:.1}), image={}x{}",
            header.naxis1, header.naxis2
        ));
    }
    let (x0, y0) = (x0 as usize, y0 as usize);

    let bytes_per_pixel = (header.bitpix.unsigned_abs() as usize) / 8;
    let row_stride = header.naxis1 * bytes_per_pixel;
    let last_byte = header.data_offset + (y0 + height) * row_stride;
    if last_byte > mmap.len() {
        return err("truncated file: fewer pixels than the header announces");
    }

    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        let row_offset = header.data_offset + (y0 + row) * row_stride + x0 * bytes_per_pixel;
        let row_bytes = &mmap[row_offset..row_offset + width * bytes_per_pixel];
        for col in 0..width {
            let off = col * bytes_per_pixel;
            let raw = &row_bytes[off..off + bytes_per_pixel];
            let v = match header.bitpix {
                16 => i16::from_be_slice(raw),
                8 => u8::from_be_slice(raw),
                32 => i32::from_be_slice(raw),
                64 => i64::from_be_slice(raw),
                -32 => f32::from_be_slice(raw),
                -64 => f64::from_be_slice(raw),
                other => return err(format!("BITPIX {other} is not supported")),
            };
            data.push(header.bzero + header.bscale * v);
        }
    }

    Ok((px, py, x0, y0, data))
}

/// Extracts a cutout (small rectangle) around sky coordinates (RA/Dec in
/// degrees) without reading the whole file: through mmap only the few
/// rows needed are touched -- the OS only loads into RAM the memory pages
/// that correspond to the bytes actually read (that is the project's whole
/// point: no need to read 1 GB to extract a thumbnail). `hdu` selects the
/// HDU by 0-based index or EXTNAME; by default the primary HDU is read.
#[pyfunction]
#[pyo3(signature = (path, ra_deg, dec_deg, width, height, hdu=None))]
fn extract_cutout(
    path: &str,
    ra_deg: f64,
    dec_deg: f64,
    width: usize,
    height: usize,
    hdu: Option<HduSelector>,
) -> PyResult<Cutout> {
    let start = std::time::Instant::now();
    let (px, py, _x0, _y0, data) = extract_cutout_raw(path, ra_deg, dec_deg, width, height, &hdu)?;
    Ok(Cutout {
        width,
        height,
        pixel_x_center: px,
        pixel_y_center: py,
        data,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Result of a Parquet export.
#[pyclass]
struct ParquetExport {
    #[pyo3(get)]
    out_path: String,
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    elapsed_ms: f64,
}

/// Extracts a cutout AND exports it directly as Parquet (modern columnar
/// format, read instantly by pandas/polars/DuckDB): one row per pixel,
/// columns (x, y, value), where x/y are the absolute pixel coordinates in
/// the source image (not just within the cutout), so that several cutouts
/// can be stitched back together if needed.
#[pyfunction]
#[pyo3(signature = (path, ra_deg, dec_deg, width, height, out_path, hdu=None))]
fn export_cutout_parquet(
    path: &str,
    ra_deg: f64,
    dec_deg: f64,
    width: usize,
    height: usize,
    out_path: &str,
    hdu: Option<HduSelector>,
) -> PyResult<ParquetExport> {
    let start = std::time::Instant::now();
    let (_px, _py, x0, y0, data) = extract_cutout_raw(path, ra_deg, dec_deg, width, height, &hdu)?;

    let n = data.len();
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for row in 0..height {
        for col in 0..width {
            xs.push((x0 + col) as i64);
            ys.push((y0 + row) as i64);
        }
    }

    let x_arr: arrow_array::ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(xs));
    let y_arr: arrow_array::ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(ys));
    let v_arr: arrow_array::ArrayRef = std::sync::Arc::new(arrow_array::Float64Array::from(data));

    let batch = arrow_array::RecordBatch::try_from_iter([("x", x_arr), ("y", y_arr), ("value", v_arr)])
        .map_err(|e| PyValueError::new_err(format!("failed to build the RecordBatch: {e}")))?;

    let out_file = File::create(out_path)
        .map_err(|e| PyValueError::new_err(format!("failed to create {out_path}: {e}")))?;
    let mut writer = parquet::arrow::arrow_writer::ArrowWriter::try_new(out_file, batch.schema(), None)
        .map_err(|e| PyValueError::new_err(format!("failed to initialise the ArrowWriter: {e}")))?;
    writer
        .write(&batch)
        .map_err(|e| PyValueError::new_err(format!("Parquet write failed: {e}")))?;
    writer
        .close()
        .map_err(|e| PyValueError::new_err(format!("Parquet close failed: {e}")))?;

    Ok(ParquetExport {
        out_path: out_path.to_string(),
        rows: n,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Number of logical cores detected on the machine (to check that the
/// hardware adaptation works, with nothing hardcoded).
#[pyfunction]
fn detected_cpus() -> usize {
    num_cpus::get()
}

/// The Python module exposed to users (`import euclid_fast`).
#[pymodule]
fn euclid_fast(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fits_stats, m)?)?;
    m.add_function(wrap_pyfunction!(extract_cutout, m)?)?;
    m.add_function(wrap_pyfunction!(export_cutout_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(list_hdus, m)?)?;
    m.add_function(wrap_pyfunction!(detected_cpus, m)?)?;
    m.add_class::<FitsStats>()?;
    m.add_class::<Cutout>()?;
    m.add_class::<ParquetExport>()?;
    m.add_class::<HduInfo>()?;
    Ok(())
}
