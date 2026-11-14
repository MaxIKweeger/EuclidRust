use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;

const CARD_LEN: usize = 80;
const BLOCK_LEN: usize = 2880; // 36 cards of 80 bytes

/// WCS (World Coordinate System) parameters: the standard FITS
/// "pixel <-> sky coordinates" transformation, using the tangent (TAN)
/// projection. Absent (`None`) if the file has no modern astrometry, e.g.
/// old plate-solution headers that use non-standard keywords.
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
}

/// Minimal FITS header: only what is needed to read the pixel array.
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

/// Parses the FITS header cards (2880-byte blocks = 36 cards of 80 bytes)
/// up to the "END" card. Only handles simple images (single HDU, NAXIS=2,
/// no extensions), which is what the supported Euclid mosaics look like.
fn parse_header(mmap: &[u8]) -> PyResult<FitsHeader> {
    let mut bitpix: Option<i32> = None;
    let mut naxis: Option<usize> = None;
    let mut ctype1: Option<String> = None;
    let mut ctype2: Option<String> = None;
    let mut naxis1: Option<usize> = None;
    let mut naxis2: Option<usize> = None;
    let mut bzero: f64 = 0.0;
    let mut bscale: f64 = 1.0;
    let mut crval1: Option<f64> = None;
    let mut crval2: Option<f64> = None;
    let mut crpix1: Option<f64> = None;
    let mut crpix2: Option<f64> = None;
    let mut cd1_1: Option<f64> = None;
    let mut cd1_2: Option<f64> = None;
    let mut cd2_1: Option<f64> = None;
    let mut cd2_2: Option<f64> = None;

    let mut block_start = 0usize;
    'blocks: loop {
        if block_start + BLOCK_LEN > mmap.len() {
            return Err(PyValueError::new_err(
                "truncated file: no END card found",
            ));
        }
        let block = &mmap[block_start..block_start + BLOCK_LEN];
        for card_idx in 0..(BLOCK_LEN / CARD_LEN) {
            let card = &block[card_idx * CARD_LEN..(card_idx + 1) * CARD_LEN];
            let card_str = std::str::from_utf8(card).unwrap_or("");
            let keyword = card_str[0..8].trim();

            if keyword == "END" {
                let data_offset = block_start + BLOCK_LEN;
                let naxis = naxis.ok_or_else(|| PyValueError::new_err("missing NAXIS"))?;
                if naxis == 0 {
                    return Err(PyValueError::new_err(
                        "the primary HDU holds no image (NAXIS = 0): the image is probably stored in an \
                         extension or is tile-compressed (e.g. .fits.fz), which is not supported yet",
                    ));
                }
                if naxis != 2 {
                    return Err(PyValueError::new_err(format!(
                        "NAXIS = {naxis} is not supported: only 2-D images are handled (3-D cubes are not)"
                    )));
                }
                let bitpix = bitpix.ok_or_else(|| PyValueError::new_err("missing BITPIX"))?;
                let naxis1 = naxis1.ok_or_else(|| PyValueError::new_err("missing NAXIS1"))?;
                let naxis2 = naxis2.ok_or_else(|| PyValueError::new_err("missing NAXIS2"))?;
                let wcs = match (crval1, crval2, crpix1, crpix2, cd1_1, cd2_2) {
                    (Some(crval1), Some(crval2), Some(crpix1), Some(crpix2), Some(cd1_1), Some(cd2_2)) => {
                        Some(Wcs {
                            crval1,
                            crval2,
                            crpix1,
                            crpix2,
                            cd1_1,
                            cd1_2: cd1_2.unwrap_or(0.0),
                            cd2_1: cd2_1.unwrap_or(0.0),
                            cd2_2,
                        })
                    }
                    _ => None,
                };
                return Ok(FitsHeader {
                    bitpix,
                    naxis1,
                    naxis2,
                    bzero,
                    bscale,
                    data_offset,
                    wcs,
                    ctype1,
                    ctype2,
                });
            }

            if card_str.len() >= 10 && &card_str[8..10] == "= " {
                let value_part = &card_str[10..];
                let value_str = value_part.split('/').next().unwrap_or("").trim();
                match keyword {
                    "BITPIX" => bitpix = value_str.parse::<i32>().ok(),
                    "NAXIS" => naxis = value_str.parse::<usize>().ok(),
                    "CTYPE1" => ctype1 = Some(value_str.trim_matches('\'').trim().to_string()),
                    "CTYPE2" => ctype2 = Some(value_str.trim_matches('\'').trim().to_string()),
                    "NAXIS1" => naxis1 = value_str.parse::<usize>().ok(),
                    "NAXIS2" => naxis2 = value_str.parse::<usize>().ok(),
                    "BZERO" => bzero = value_str.parse::<f64>().unwrap_or(0.0),
                    "BSCALE" => bscale = value_str.parse::<f64>().unwrap_or(1.0),
                    "CRVAL1" => crval1 = value_str.parse::<f64>().ok(),
                    "CRVAL2" => crval2 = value_str.parse::<f64>().ok(),
                    "CRPIX1" => crpix1 = value_str.parse::<f64>().ok(),
                    "CRPIX2" => crpix2 = value_str.parse::<f64>().ok(),
                    "CD1_1" => cd1_1 = value_str.parse::<f64>().ok(),
                    "CD1_2" => cd1_2 = value_str.parse::<f64>().ok(),
                    "CD2_1" => cd2_1 = value_str.parse::<f64>().ok(),
                    "CD2_2" => cd2_2 = value_str.parse::<f64>().ok(),
                    _ => {}
                }
            }
        }
        block_start += BLOCK_LEN;
        if block_start > 50 * BLOCK_LEN {
            // Safety guard: a sane FITS header is never this long.
            break 'blocks;
        }
    }
    Err(PyValueError::new_err("END card not found"))
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

/// Reads a FITS file (simple, uncompressed image) through mmap (zero-copy
/// for the disk -> memory part) and computes min/max/mean in parallel with
/// rayon. The rayon thread pool sizes itself to the number of cores
/// available on the machine (no hardcoded value).
#[pyfunction]
fn fits_stats(path: &str) -> PyResult<FitsStats> {
    let start = std::time::Instant::now();

    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("cannot open {path}: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| PyValueError::new_err(format!("mmap failed: {e}")))?
    };

    let header = parse_header(&mmap)?;
    let n_pixels = header.naxis1 * header.naxis2;
    let bytes_per_pixel = (header.bitpix.unsigned_abs() as usize) / 8;
    let data_end = header.data_offset + n_pixels * bytes_per_pixel;

    if data_end > mmap.len() {
        return Err(PyValueError::new_err(
            "truncated file: fewer pixels than the header announces",
        ));
    }
    let data = &mmap[header.data_offset..data_end];

    // Read + convert to "physical" f64 (BZERO + BSCALE * raw_value), in
    // parallel over chunks of pixels. rayon sizes the number of threads
    // itself from the machine's logical core count.
    let (min, max, sum) = match header.bitpix {
        16 => reduce_be::<i16>(data, header.bzero, header.bscale),
        8 => reduce_be::<u8>(data, header.bzero, header.bscale),
        32 => reduce_be::<i32>(data, header.bzero, header.bscale),
        -32 => reduce_be::<f32>(data, header.bzero, header.bscale),
        -64 => reduce_be::<f64>(data, header.bzero, header.bscale),
        other => {
            return Err(PyValueError::new_err(format!(
                "BITPIX {other} is not supported yet"
            )))
        }
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

/// Converts sky coordinates (RA/Dec, in degrees) to pixel coordinates (FITS
/// 1-indexed convention) using the standard tangent (TAN / gnomonic)
/// projection -- the projection used by almost all modern surveys
/// (Euclid included) for images that are not too wide-field.
/// SIP distortion (higher-order terms) is not handled: fine for a first
/// version, to be refined if real Euclid images need it.
fn radec_to_pixel(ra_deg: f64, dec_deg: f64, wcs: &Wcs) -> (f64, f64) {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let ra0 = wcs.crval1.to_radians();
    let dec0 = wcs.crval2.to_radians();

    let cos_c = dec0.sin() * dec.sin() + dec0.cos() * dec.cos() * (ra - ra0).cos();
    let xi = dec.cos() * (ra - ra0).sin() / cos_c; // radians
    let eta = (dec0.cos() * dec.sin() - dec0.sin() * dec.cos() * (ra - ra0).cos()) / cos_c;

    let xi_deg = xi.to_degrees();
    let eta_deg = eta.to_degrees();

    // Invert the 2x2 CD matrix to go from intermediate world coordinates
    // (xi, eta) to pixel offsets.
    let det = wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1;
    let dx = (wcs.cd2_2 * xi_deg - wcs.cd1_2 * eta_deg) / det;
    let dy = (-wcs.cd2_1 * xi_deg + wcs.cd1_1 * eta_deg) / det;

    (wcs.crpix1 + dx, wcs.crpix2 + dy)
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
) -> PyResult<(f64, f64, usize, usize, Vec<f64>)> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("cannot open {path}: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| PyValueError::new_err(format!("mmap failed: {e}")))?
    };
    let header = parse_header(&mmap)?;
    let wcs = header
        .wcs
        .ok_or_else(|| PyValueError::new_err("this file has no WCS header (CRVAL/CRPIX/CD)"))?;

    // Only the plain tangent projection is implemented. Anything else (SIN, ARC,
    // TAN-SIP with distortion terms...) would silently give wrong positions.
    if header.ctype1.as_deref() != Some("RA---TAN") || header.ctype2.as_deref() != Some("DEC--TAN") {
        return Err(PyValueError::new_err(format!(
            "unsupported WCS projection: expected CTYPE1/CTYPE2 = RA---TAN/DEC--TAN, found {}/{} \
             (other projections and SIP distortion are not supported)",
            header.ctype1.as_deref().unwrap_or("<missing>"),
            header.ctype2.as_deref().unwrap_or("<missing>"),
        )));
    }

    let (px, py) = radec_to_pixel(ra_deg, dec_deg, &wcs);
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
        return Err(PyValueError::new_err(format!(
            "cutout outside the image: center pixel=({px:.1},{py:.1}), image={}x{}",
            header.naxis1, header.naxis2
        )));
    }
    let (x0, y0) = (x0 as usize, y0 as usize);

    let bytes_per_pixel = (header.bitpix.unsigned_abs() as usize) / 8;
    let row_stride = header.naxis1 * bytes_per_pixel;

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
                -32 => f32::from_be_slice(raw),
                -64 => f64::from_be_slice(raw),
                other => return Err(PyValueError::new_err(format!("BITPIX {other} is not supported"))),
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
/// point: no need to read 1 GB to extract a thumbnail).
#[pyfunction]
fn extract_cutout(path: &str, ra_deg: f64, dec_deg: f64, width: usize, height: usize) -> PyResult<Cutout> {
    let start = std::time::Instant::now();
    let (px, py, _x0, _y0, data) = extract_cutout_raw(path, ra_deg, dec_deg, width, height)?;
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
fn export_cutout_parquet(
    path: &str,
    ra_deg: f64,
    dec_deg: f64,
    width: usize,
    height: usize,
    out_path: &str,
) -> PyResult<ParquetExport> {
    let start = std::time::Instant::now();
    let (_px, _py, x0, y0, data) = extract_cutout_raw(path, ra_deg, dec_deg, width, height)?;

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
    m.add_function(wrap_pyfunction!(detected_cpus, m)?)?;
    m.add_class::<FitsStats>()?;
    m.add_class::<Cutout>()?;
    m.add_class::<ParquetExport>()?;
    Ok(())
}
