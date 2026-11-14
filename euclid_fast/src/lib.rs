use flate2::read::GzDecoder;
use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::sync::OnceLock;

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

/// The zenithal projections (the reference point is the native pole), which
/// differ only by how the radius R on the plane depends on the angle from the
/// pole. TPV is TAN followed by a polynomial (see `Tpv`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Proj {
    Tan, // gnomonic
    Sin, // orthographic
    Arc, // zenithal equidistant
    Stg, // stereographic
    Zea, // zenithal equal-area
    Zpn, // zenithal polynomial (coefficients PV2_m)
}

/// SIP distortion (Shupe et al. 2005): polynomials A and B in the pixel offsets
/// (u, v) from CRPIX, applied before the CD matrix. Each term is (p, q, coef)
/// for coef * u^p * v^q.
#[derive(Debug, Clone)]
struct Sip {
    a: Vec<(i32, i32, f64)>,
    b: Vec<(i32, i32, f64)>,
}

/// WCS (World Coordinate System) parameters: the standard FITS
/// "pixel <-> sky coordinates" transformation for a zenithal projection,
/// optionally with TPV or SIP distortion. Absent (`None`) if the header has no
/// usable astrometry keywords.
#[derive(Debug, Clone)]
struct Wcs {
    crval1: f64, // RA of the reference pixel, in degrees
    crval2: f64, // Dec of the reference pixel, in degrees
    crpix1: f64, // reference pixel, axis 1 (FITS convention, 1-indexed)
    crpix2: f64, // reference pixel, axis 2
    cd: [f64; 4], // CD1_1, CD1_2, CD2_1, CD2_2: linear matrix, degrees/pixel
    proj: Proj,
    lonpole: f64,      // native longitude of the celestial pole, degrees
    zpn: [f64; N_PV],  // PV2_m coefficients (ZPN only)
    tpv: Option<Tpv>,  // Some(..) when CTYPE is RA---TPV / DEC--TPV
    sip: Option<Sip>,  // Some(..) when CTYPE is RA---TAN-SIP / DEC--TAN-SIP
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
    cdelt1: Option<f64>,
    cdelt2: Option<f64>,
    pc: [Option<f64>; 4], // PC1_1, PC1_2, PC2_1, PC2_2
    crota2: Option<f64>,
    lonpole: Option<f64>,
    sip_a: Vec<(i32, i32, f64)>,
    sip_b: Vec<(i32, i32, f64)>,
    // --- tile-compressed images (a BINTABLE with ZIMAGE = T) ---
    zbitpix: Option<i32>,
    znaxis: Option<usize>,
    znaxes: Vec<usize>, // ZNAXIS1, ZNAXIS2, ...
    ztiles: Vec<usize>, // ZTILE1, ZTILE2, ...
    zcmptype: Option<String>,
    znames: Vec<Option<String>>, // ZNAME1 .. ZNAME9
    zvals: Vec<Option<f64>>,     // ZVAL1 .. ZVAL9
    zquantiz: Option<String>,
    zdither0: Option<i64>,
    zblank: Option<i64>,
    theap: Option<usize>,
    tfields: usize,
    ttype: Vec<String>, // TTYPE1 ..
    tform: Vec<String>, // TFORM1 ..
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
            cdelt1: None,
            cdelt2: None,
            pc: [None; 4],
            crota2: None,
            lonpole: None,
            sip_a: Vec::new(),
            sip_b: Vec::new(),
            zbitpix: None,
            znaxis: None,
            znaxes: Vec::new(),
            ztiles: Vec::new(),
            zcmptype: None,
            znames: vec![None; 9],
            zvals: vec![None; 9],
            zquantiz: None,
            zdither0: None,
            zblank: None,
            theap: None,
            tfields: 0,
            ttype: Vec::new(),
            tform: Vec::new(),
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
    data_offset: usize, // byte offset where the pixels start (raw images)
    wcs: Option<Wcs>,
    tiled: Option<Tiled>,   // Some(..) for tile-compressed images
    wcs_error: Option<String>, // why the WCS cannot be used, if it is present but unsupported
}

fn parse_f64(s: &str) -> Option<f64> {
    // FITS allows a Fortran-style "D" exponent (1.0D+00)
    s.replace(['D', 'd'], "E").parse::<f64>().ok()
}

fn unquote(s: &str) -> String {
    s.trim_matches('\'').trim().to_string()
}

/// "2_1" -> (2, 1), the powers of a SIP term (A_2_1).
fn parse_pq(s: &str) -> Option<(i32, i32)> {
    let (p, q) = s.split_once('_')?;
    Some((p.parse().ok()?, q.parse().ok()?))
}

fn set_indexed<T: Clone>(v: &mut Vec<T>, i: usize, value: T, fill: T) {
    if v.len() < i {
        v.resize(i, fill);
    }
    v[i - 1] = value;
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
                    "THEAP" => h.theap = value_str.parse::<usize>().ok(),
                    "TFIELDS" => h.tfields = value_str.parse::<usize>().unwrap_or(0),
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
                    "CDELT1" => h.cdelt1 = parse_f64(value_str),
                    "CDELT2" => h.cdelt2 = parse_f64(value_str),
                    "PC1_1" => h.pc[0] = parse_f64(value_str),
                    "PC1_2" => h.pc[1] = parse_f64(value_str),
                    "PC2_1" => h.pc[2] = parse_f64(value_str),
                    "PC2_2" => h.pc[3] = parse_f64(value_str),
                    "CROTA2" => h.crota2 = parse_f64(value_str),
                    "LONPOLE" => h.lonpole = parse_f64(value_str),
                    "CTYPE1" => h.ctype1 = Some(unquote(value_str)),
                    "CTYPE2" => h.ctype2 = Some(unquote(value_str)),
                    "ZBITPIX" => h.zbitpix = value_str.parse::<i32>().ok(),
                    "ZNAXIS" => h.znaxis = value_str.parse::<usize>().ok(),
                    "ZCMPTYPE" => h.zcmptype = Some(unquote(value_str)),
                    "ZQUANTIZ" => h.zquantiz = Some(unquote(value_str)),
                    "ZDITHER0" => h.zdither0 = value_str.parse::<i64>().ok(),
                    "ZBLANK" => h.zblank = value_str.parse::<i64>().ok(),
                    _ => {
                        let idx = |rest: &str| rest.parse::<usize>().ok().filter(|i| (1..=999).contains(i));
                        if let Some(rest) = keyword.strip_prefix("NAXIS") {
                            if let Some(i) = idx(rest) {
                                set_indexed(&mut h.naxes, i, value_str.parse::<usize>().unwrap_or(0), 0);
                            }
                        } else if let Some(rest) = keyword.strip_prefix("ZNAXIS") {
                            if let Some(i) = idx(rest) {
                                set_indexed(&mut h.znaxes, i, value_str.parse::<usize>().unwrap_or(0), 0);
                            }
                        } else if let Some(rest) = keyword.strip_prefix("ZTILE") {
                            if let Some(i) = idx(rest) {
                                set_indexed(&mut h.ztiles, i, value_str.parse::<usize>().unwrap_or(0), 0);
                            }
                        } else if let Some(rest) = keyword.strip_prefix("ZNAME") {
                            if let Some(i) = idx(rest).filter(|i| *i <= 9) {
                                h.znames[i - 1] = Some(unquote(value_str));
                            }
                        } else if let Some(rest) = keyword.strip_prefix("ZVAL") {
                            if let Some(i) = idx(rest).filter(|i| *i <= 9) {
                                h.zvals[i - 1] = parse_f64(value_str);
                            }
                        } else if let Some(rest) = keyword.strip_prefix("TTYPE") {
                            if let Some(i) = idx(rest) {
                                set_indexed(&mut h.ttype, i, unquote(value_str), String::new());
                            }
                        } else if let Some(rest) = keyword.strip_prefix("TFORM") {
                            if let Some(i) = idx(rest) {
                                set_indexed(&mut h.tform, i, unquote(value_str), String::new());
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
                        } else if let Some(rest) = keyword.strip_prefix("A_") {
                            if let (Some((p, q)), Some(v)) = (parse_pq(rest), parse_f64(value_str)) {
                                h.sip_a.push((p, q, v));
                            }
                        } else if let Some(rest) = keyword.strip_prefix("B_") {
                            if let (Some((p, q)), Some(v)) = (parse_pq(rest), parse_f64(value_str)) {
                                h.sip_b.push((p, q, v));
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
        match build_tiled(h, hdu.data_offset) {
            Ok(t) => format!(
                "[{}]{} (tile-compressed {}, {}x{}, BITPIX {})",
                hdu.index,
                label,
                t.algo.name(),
                t.naxis1,
                t.naxis2,
                t.zbitpix
            ),
            Err(_) => format!(
                "[{}]{} (tile-compressed with {}, not supported yet)",
                hdu.index,
                label,
                h.zcmptype.clone().unwrap_or_else(|| "unknown algorithm".to_string())
            ),
        }
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

/// The 2x2 linear matrix (degrees/pixel) of a header: the CD matrix if there
/// is one, otherwise CDELT combined with PC (or with the legacy CROTA2).
fn linear_matrix(h: &RawHeader) -> Option<[f64; 4]> {
    if let (Some(cd11), Some(cd22)) = (h.cd1_1, h.cd2_2) {
        return Some([cd11, h.cd1_2.unwrap_or(0.0), h.cd2_1.unwrap_or(0.0), cd22]);
    }
    let has_pc = h.pc.iter().any(|v| v.is_some());
    if h.cdelt1.is_none() && h.cdelt2.is_none() && !has_pc {
        return None;
    }
    let (c1, c2) = (h.cdelt1.unwrap_or(1.0), h.cdelt2.unwrap_or(1.0));
    if !has_pc {
        if let Some(rho) = h.crota2 {
            // legacy AIPS rotation convention
            let (s, c) = rho.to_radians().sin_cos();
            return Some([c1 * c, -c2 * s, c1 * s, c2 * c]);
        }
    }
    let pc = [h.pc[0].unwrap_or(1.0), h.pc[1].unwrap_or(0.0), h.pc[2].unwrap_or(0.0), h.pc[3].unwrap_or(1.0)];
    Some([c1 * pc[0], c1 * pc[1], c2 * pc[2], c2 * pc[3]])
}

/// Recognises the projection named by CTYPE1/CTYPE2 ("RA---TAN", "DEC--TAN",
/// "RA---TAN-SIP", "RA---TPV"...). Returns (projection, tpv, sip).
fn parse_projection(h: &RawHeader) -> Result<(Proj, bool, bool), String> {
    let c1 = h.ctype1.as_deref();
    let c2 = h.ctype2.as_deref();
    let split = |c: Option<&str>, prefix: &str| -> Option<(String, String)> {
        let rest = c?.strip_prefix(prefix)?;
        Some((rest.get(0..3)?.to_string(), rest.get(3..)?.to_string()))
    };
    if let (Some((code1, suf1)), Some((code2, suf2))) = (split(c1, "RA---"), split(c2, "DEC--")) {
        if code1 == code2 && suf1 == suf2 {
            match (code1.as_str(), suf1.as_str()) {
                ("TAN", "") => return Ok((Proj::Tan, false, false)),
                ("TAN", "-SIP") => return Ok((Proj::Tan, false, true)),
                ("TPV", "") => return Ok((Proj::Tan, true, false)),
                ("SIN", "") => return Ok((Proj::Sin, false, false)),
                ("ARC", "") => return Ok((Proj::Arc, false, false)),
                ("STG", "") => return Ok((Proj::Stg, false, false)),
                ("ZEA", "") => return Ok((Proj::Zea, false, false)),
                ("ZPN", "") => return Ok((Proj::Zpn, false, false)),
                _ => {}
            }
        }
    }
    Err(format!(
        "unsupported WCS projection: found CTYPE1/CTYPE2 = {}/{}; supported are RA---TAN/DEC--TAN (also with -SIP \
         distortion), RA---TPV/DEC--TPV and the zenithal projections SIN, ARC, STG, ZEA and ZPN \
         (other projections, for example cylindrical or all-sky ones, are not supported)",
        c1.unwrap_or("<missing>"),
        c2.unwrap_or("<missing>")
    ))
}

/// Builds the WCS of a header. Returns (wcs, problem): `wcs` is `None` when the
/// astrometry keywords are missing or the projection is not supported, and in
/// the second case `problem` says why.
fn build_wcs(h: &RawHeader) -> (Option<Wcs>, Option<String>) {
    let (crval1, crval2, crpix1, crpix2, cd) = match (h.crval1, h.crval2, h.crpix1, h.crpix2, linear_matrix(h)) {
        (Some(a), Some(b), Some(c), Some(d), Some(m)) => (a, b, c, d, m),
        _ => return (None, None),
    };
    let (proj, is_tpv, is_sip) = match parse_projection(h) {
        Ok(v) => v,
        Err(msg) => return (None, Some(msg)),
    };
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
    let tpv = if is_tpv { Some(Tpv { pv1: coeffs(&h.pv1), pv2: coeffs(&h.pv2) }) } else { None };
    let sip = if is_sip { Some(Sip { a: h.sip_a.clone(), b: h.sip_b.clone() }) } else { None };
    // Default LONPOLE for a zenithal projection (native pole = reference point):
    // 180 degrees unless the reference point is at the celestial pole.
    let lonpole = h.lonpole.unwrap_or(if crval2 >= 90.0 { 0.0 } else { 180.0 });
    (
        Some(Wcs { crval1, crval2, crpix1, crpix2, cd, proj, lonpole, zpn: coeffs(&h.pv2), tpv, sip }),
        None,
    )
}

// ---------------------------------------------------------------------------
// Tile-compressed images (FITS "tiled image compression": the image is cut
// into tiles, each compressed on its own and stored in a row of a BINTABLE).
// Supported: RICE_1, GZIP_1, GZIP_2, with optional float quantization
// (NO_DITHER, SUBTRACTIVE_DITHER_1, SUBTRACTIVE_DITHER_2).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Algo {
    Rice,
    Gzip1,
    Gzip2,
}

impl Algo {
    fn name(&self) -> &'static str {
        match self {
            Algo::Rice => "RICE_1",
            Algo::Gzip1 => "GZIP_1",
            Algo::Gzip2 => "GZIP_2",
        }
    }
}

/// A column of the compressed-image BINTABLE.
#[derive(Debug, Clone, Copy)]
struct Column {
    offset: usize, // byte offset inside a table row
    letter: char,  // TFORM type letter: 'P'/'Q' (variable length), 'D', 'E', 'J', ...
    wide: bool,    // 'Q' descriptors (64-bit) instead of 'P' (32-bit)
    elem: char,    // element type of a variable-length column ('B', 'I', 'J'...)
}

/// What is needed to decode the tiles of a compressed image.
#[derive(Debug, Clone)]
struct Tiled {
    zbitpix: i32,
    naxis1: usize,
    naxis2: usize,
    tile1: usize, // tile width in pixels
    tile2: usize, // tile height in pixels
    algo: Algo,
    blocksize: usize, // Rice block size (ZVAL1, default 32)
    bytepix: usize,   // Rice bytes per integer (ZVAL2, default 4)
    dither: u8,       // 0 = no dither, 1 = SUBTRACTIVE_DITHER_1, 2 = SUBTRACTIVE_DITHER_2
    dither_seed: i64, // ZDITHER0
    zblank: Option<i64>,
    bzero: f64,
    bscale: f64,
    table_offset: usize, // where the BINTABLE rows start
    row_len: usize,
    nrows: usize,
    heap_offset: usize, // where the variable-length data (the heap) starts
    comp: Option<Column>,
    gzip: Option<Column>,
    uncomp: Option<Column>,
    zscale: Option<Column>,
    zzero: Option<Column>,
    zblank_col: Option<Column>,
}

fn parse_columns(h: &RawHeader) -> Result<Vec<(String, Column)>, String> {
    let mut cols = Vec::new();
    let mut offset = 0usize;
    for i in 0..h.tfields {
        let name = h.ttype.get(i).cloned().unwrap_or_default();
        let form = h.tform.get(i).cloned().unwrap_or_default();
        let digits: String = form.chars().take_while(|c| c.is_ascii_digit()).collect();
        let repeat: usize = if digits.is_empty() { 1 } else { digits.parse().unwrap_or(1) };
        let mut rest = form[digits.len()..].chars();
        let letter = rest.next().ok_or_else(|| format!("bad TFORM{} in the compressed-image table", i + 1))?;
        let elem = rest.next().unwrap_or('B');
        let size = match letter {
            'L' | 'B' | 'A' => 1,
            'X' => repeat.div_ceil(8),
            'I' => 2 * repeat,
            'J' | 'E' => 4 * repeat,
            'K' | 'D' | 'C' => 8 * repeat,
            'M' => 16 * repeat,
            'P' => 8 * repeat,
            'Q' => 16 * repeat,
            other => return Err(format!("unsupported column type '{other}' in the compressed-image table")),
        };
        let size = if matches!(letter, 'L' | 'B' | 'A') { repeat } else { size };
        cols.push((name, Column { offset, letter, wide: letter == 'Q', elem }));
        offset += size;
    }
    Ok(cols)
}

fn build_tiled(h: &RawHeader, data_offset: usize) -> Result<Tiled, String> {
    let ztype = h.zcmptype.clone().unwrap_or_default();
    let algo = match ztype.as_str() {
        "RICE_1" | "RICE_ONE" => Algo::Rice,
        "GZIP_1" => Algo::Gzip1,
        "GZIP_2" => Algo::Gzip2,
        other => {
            return Err(format!(
                "tile-compressed image uses ZCMPTYPE = {} which is not supported yet (supported: RICE_1, GZIP_1, GZIP_2)",
                if other.is_empty() { "<missing>" } else { other }
            ))
        }
    };
    let zbitpix = h.zbitpix.ok_or("missing ZBITPIX in the compressed image header")?;
    if !matches!(zbitpix, 8 | 16 | 32 | 64 | -32 | -64) {
        return Err(format!("ZBITPIX {zbitpix} is not supported"));
    }
    let znaxis = h.znaxis.ok_or("missing ZNAXIS in the compressed image header")?;
    if znaxis != 2 {
        return Err(format!(
            "ZNAXIS = {znaxis} is not supported: only 2-D images are handled (3-D cubes are not)"
        ));
    }
    let naxis1 = h.znaxes.first().copied().filter(|&v| v > 0).ok_or("missing ZNAXIS1")?;
    let naxis2 = h.znaxes.get(1).copied().filter(|&v| v > 0).ok_or("missing ZNAXIS2")?;
    let tile1 = h.ztiles.first().copied().filter(|&v| v > 0).unwrap_or(naxis1);
    let tile2 = h.ztiles.get(1).copied().filter(|&v| v > 0).unwrap_or(1);

    let mut blocksize = 32usize;
    let mut bytepix = 4usize;
    for (name, val) in h.znames.iter().zip(h.zvals.iter()) {
        match (name.as_deref(), val) {
            (Some(n), Some(v)) if n.eq_ignore_ascii_case("BLOCKSIZE") => blocksize = *v as usize,
            (Some(n), Some(v)) if n.eq_ignore_ascii_case("BYTEPIX") => bytepix = *v as usize,
            _ => {}
        }
    }
    if algo == Algo::Rice && !matches!(bytepix, 1 | 2 | 4) {
        return Err(format!("Rice BYTEPIX = {bytepix} is not supported (1, 2 or 4)"));
    }
    if blocksize == 0 {
        return Err("Rice BLOCKSIZE = 0".to_string());
    }
    let dither = match h.zquantiz.as_deref() {
        None | Some("NO_DITHER") | Some("NONE") => 0u8,
        Some("SUBTRACTIVE_DITHER_1") => 1,
        Some("SUBTRACTIVE_DITHER_2") => 2,
        Some(other) => return Err(format!("ZQUANTIZ = {other} is not supported")),
    };

    // The BINTABLE geometry: NAXIS1 = bytes per row, NAXIS2 = number of rows = tiles.
    let row_len = h.naxes.first().copied().ok_or("missing NAXIS1 in the compressed image table")?;
    let nrows = h.naxes.get(1).copied().ok_or("missing NAXIS2 in the compressed image table")?;
    let expected_tiles = naxis1.div_ceil(tile1) * naxis2.div_ceil(tile2);
    if nrows != expected_tiles {
        return Err(format!(
            "the compressed-image table has {nrows} rows but the tiling implies {expected_tiles} tiles"
        ));
    }
    let heap_offset = data_offset + h.theap.unwrap_or(row_len * nrows);

    let cols = parse_columns(h)?;
    let find = |name: &str| cols.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, c)| *c);
    let comp = find("COMPRESSED_DATA");
    let gzip = find("GZIP_COMPRESSED_DATA");
    let uncomp = find("UNCOMPRESSED_DATA");
    if comp.is_none() && gzip.is_none() && uncomp.is_none() {
        return Err("the compressed-image table has no COMPRESSED_DATA column".to_string());
    }
    for c in [comp, gzip, uncomp].into_iter().flatten() {
        if matches!(c.letter, 'P' | 'Q') && c.elem != 'B' {
            return Err(format!("unsupported variable-length element type '{}' for tile data", c.elem));
        }
    }

    Ok(Tiled {
        zbitpix,
        naxis1,
        naxis2,
        tile1,
        tile2,
        algo,
        blocksize,
        bytepix,
        dither,
        dither_seed: h.zdither0.unwrap_or(1),
        zblank: h.zblank,
        bzero: h.bzero,
        bscale: h.bscale,
        table_offset: data_offset,
        row_len,
        nrows,
        heap_offset,
        comp,
        gzip,
        uncomp,
        zscale: find("ZSCALE"),
        zzero: find("ZZERO"),
        zblank_col: find("ZBLANK"),
    })
}

fn be_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}
fn be_u64(b: &[u8]) -> u64 {
    u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// The bytes a variable-length column points to for table row `row`.
fn heap_bytes<'a>(t: &Tiled, mmap: &'a [u8], col: &Column, row: usize) -> PyResult<&'a [u8]> {
    let base = t.table_offset + row * t.row_len + col.offset;
    let width = if col.wide { 16 } else { 8 };
    if base + width > mmap.len() {
        return err("truncated file: the compressed-image table is cut short");
    }
    let (count, off) = if col.wide {
        (be_u64(&mmap[base..]) as usize, be_u64(&mmap[base + 8..]) as usize)
    } else {
        (be_u32(&mmap[base..]) as usize, be_u32(&mmap[base + 4..]) as usize)
    };
    let start = t.heap_offset + off;
    let end = start + count;
    if end > mmap.len() {
        return err("truncated file: a compressed tile points past the end of the file");
    }
    Ok(&mmap[start..end])
}

fn column_f64(t: &Tiled, mmap: &[u8], col: &Column, row: usize) -> PyResult<f64> {
    let base = t.table_offset + row * t.row_len + col.offset;
    match col.letter {
        'D' if base + 8 <= mmap.len() => Ok(f64::from_bits(be_u64(&mmap[base..]))),
        'E' if base + 4 <= mmap.len() => Ok(f32::from_bits(be_u32(&mmap[base..])) as f64),
        _ => err("unsupported or truncated ZSCALE/ZZERO column in the compressed-image table"),
    }
}

fn column_i64(t: &Tiled, mmap: &[u8], col: &Column, row: usize) -> Option<i64> {
    let base = t.table_offset + row * t.row_len + col.offset;
    match col.letter {
        'J' if base + 4 <= mmap.len() => Some(be_u32(&mmap[base..]) as i32 as i64),
        'K' if base + 8 <= mmap.len() => Some(be_u64(&mmap[base..]) as i64),
        'I' if base + 2 <= mmap.len() => Some(i16::from_be_bytes([mmap[base], mmap[base + 1]]) as i64),
        _ => None,
    }
}

/// Rice decompression (the algorithm of cfitsio's `fits_rdecomp*`): decodes
/// `n` integers of `bytepix` bytes each.
fn rice_decode(bytes: &[u8], n: usize, blocksize: usize, bytepix: usize) -> PyResult<Vec<i64>> {
    let (fsbits, fsmax, bbits): (i64, i64, i64) = match bytepix {
        4 => (5, 25, 32),
        2 => (4, 14, 16),
        _ => (3, 6, 8),
    };
    let mask_bits: u64 = if bbits == 64 { u64::MAX } else { (1u64 << bbits) - 1 };
    let corrupt = || PyValueError::new_err("corrupt Rice-compressed tile (ran out of data)");

    let mut pos = 0usize;
    if bytes.len() < bytepix + 1 {
        return Err(corrupt());
    }
    // first value, stored raw, big-endian
    let mut lastpix: u64 = 0;
    for _ in 0..bytepix {
        lastpix = (lastpix << 8) | bytes[pos] as u64;
        pos += 1;
    }
    let next = |pos: &mut usize| -> PyResult<u64> {
        let v = *bytes.get(*pos).ok_or_else(corrupt)? as u64;
        *pos += 1;
        Ok(v)
    };

    let mut b: u64 = next(&mut pos)?;
    let mut nbits: i64 = 8;
    let mut out: Vec<i64> = Vec::with_capacity(n);
    let to_value = |v: u64| -> i64 {
        match bytepix {
            4 => v as u32 as i32 as i64,
            2 => v as u16 as i16 as i64,
            _ => v as u8 as i64,
        }
    };

    let mut i = 0usize;
    while i < n {
        nbits -= fsbits;
        while nbits < 0 {
            b = (b << 8) | next(&mut pos)?;
            nbits += 8;
        }
        let fs = (b >> nbits) as i64 - 1;
        b &= (1u64 << nbits) - 1;
        let imax = (i + blocksize).min(n);

        if fs < 0 {
            // low entropy: the whole block repeats the previous pixel
            while i < imax {
                out.push(to_value(lastpix));
                i += 1;
            }
        } else if fs == fsmax {
            // high entropy: the differences are stored raw, bbits bits each
            while i < imax {
                let mut k = bbits - nbits;
                let mut diff: u64 = b << k;
                k -= 8;
                while k >= 0 {
                    b = next(&mut pos)?;
                    diff |= b << k;
                    k -= 8;
                }
                if nbits > 0 {
                    b = next(&mut pos)?;
                    diff |= b >> (-k);
                    b &= (1u64 << nbits) - 1;
                } else {
                    b = 0;
                }
                diff &= mask_bits;
                // undo the zig-zag mapping
                let d = if diff & 1 == 0 { diff >> 1 } else { !(diff >> 1) };
                lastpix = d.wrapping_add(lastpix) & mask_bits;
                out.push(to_value(lastpix));
                i += 1;
            }
        } else {
            // normal: unary-coded high part + fs low bits
            while i < imax {
                while b == 0 {
                    nbits += 8;
                    b = next(&mut pos)?;
                }
                let bit_len = 64 - b.leading_zeros() as i64;
                let nzero = nbits - bit_len;
                nbits -= nzero + 1;
                b ^= 1u64 << nbits; // drop the leading one-bit
                nbits -= fs;
                while nbits < 0 {
                    b = (b << 8) | next(&mut pos)?;
                    nbits += 8;
                }
                let diff = ((nzero as u64) << fs) | (b >> nbits);
                b &= (1u64 << nbits) - 1;
                let d = if diff & 1 == 0 { diff >> 1 } else { !(diff >> 1) };
                lastpix = d.wrapping_add(lastpix) & mask_bits;
                out.push(to_value(lastpix));
                i += 1;
            }
        }
    }
    Ok(out)
}

fn gunzip(bytes: &[u8], expected: usize) -> PyResult<Vec<u8>> {
    let mut out = Vec::with_capacity(expected);
    GzDecoder::new(bytes)
        .read_to_end(&mut out)
        .map_err(|e| PyValueError::new_err(format!("corrupt gzip-compressed tile: {e}")))?;
    Ok(out)
}

/// Undoes the GZIP_2 byte shuffle: the compressor stores all the most
/// significant bytes first, then the next ones, and so on.
fn unshuffle(buf: &[u8], n: usize, size: usize) -> Vec<u8> {
    let mut out = vec![0u8; buf.len()];
    for i in 0..n {
        for j in 0..size {
            out[i * size + j] = buf[j * n + i];
        }
    }
    out
}

/// One big-endian integer sample of `size` bytes (1 = unsigned byte).
fn be_int(b: &[u8], size: usize) -> i64 {
    match size {
        1 => b[0] as i64,
        2 => i16::from_be_bytes([b[0], b[1]]) as i64,
        4 => i32::from_be_bytes([b[0], b[1], b[2], b[3]]) as i64,
        _ => i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
    }
}

const N_RANDOM: usize = 10000;
const ZERO_VALUE: i64 = -2147483646;

/// The 10000 pseudo-random numbers used by subtractive dithering (a
/// Park-Miller generator, exactly as in cfitsio so the dither can be undone).
fn random_table() -> &'static Vec<f32> {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let a = 16807.0f64;
        let m = 2147483647.0f64;
        let mut seed = 1.0f64;
        (0..N_RANDOM)
            .map(|_| {
                let temp = a * seed;
                seed = temp - m * ((temp / m) as i64 as f64);
                (seed / m) as f32
            })
            .collect()
    })
}

/// Samples stored as plain big-endian values (uncompressed or losslessly
/// gzipped tiles), converted to physical f64 values.
fn samples_to_f64(t: &Tiled, buf: &[u8], n: usize) -> PyResult<Vec<f64>> {
    let size = (t.zbitpix.unsigned_abs() as usize) / 8;
    if buf.len() < n * size {
        return err("corrupt tile: fewer bytes than the tile size implies");
    }
    Ok((0..n)
        .map(|i| {
            let b = &buf[i * size..(i + 1) * size];
            match t.zbitpix {
                -32 => f32::from_bits(be_u32(b)) as f64,
                -64 => f64::from_bits(be_u64(b)),
                _ => t.bzero + t.bscale * be_int(b, size) as f64,
            }
        })
        .collect())
}

/// Decodes one tile (identified by its row number in the table) into
/// physical f64 values, row-major, `w * h` of them.
fn decode_tile(t: &Tiled, mmap: &[u8], tile_index: usize) -> PyResult<Vec<f64>> {
    let ntx = t.naxis1.div_ceil(t.tile1);
    let (tx, ty) = (tile_index % ntx, tile_index / ntx);
    let w = t.tile1.min(t.naxis1 - tx * t.tile1);
    let h = t.tile2.min(t.naxis2 - ty * t.tile2);
    let n = w * h;

    let nonempty = |c: &Option<Column>| -> PyResult<Option<&[u8]>> {
        match c {
            Some(col) => Ok(Some(heap_bytes(t, mmap, col, tile_index)?).filter(|b| !b.is_empty())),
            None => Ok(None),
        }
    };

    if let Some(bytes) = nonempty(&t.comp)? {
        // A floating-point image is stored as quantized integers when the table
        // has ZSCALE / ZZERO columns; without them (lossless mode) cfitsio puts
        // the gzipped raw floating-point samples in this same column.
        let is_quantized = t.zbitpix < 0 && t.zscale.is_some();
        if t.zbitpix < 0 && !is_quantized {
            if t.algo == Algo::Rice {
                return err("Rice-compressed floating-point tile without a ZSCALE column");
            }
            let size = (t.zbitpix.unsigned_abs() as usize) / 8;
            let mut buf = gunzip(bytes, n * size)?;
            if t.algo == Algo::Gzip2 {
                buf = unshuffle(&buf, n, size);
            }
            return samples_to_f64(t, &buf, n);
        }
        let int_size = if is_quantized { 4 } else { (t.zbitpix.unsigned_abs() as usize) / 8 };
        let ints: Vec<i64> = match t.algo {
            Algo::Rice => rice_decode(bytes, n, t.blocksize, t.bytepix)?,
            Algo::Gzip1 | Algo::Gzip2 => {
                let mut buf = gunzip(bytes, n * int_size)?;
                if buf.len() != n * int_size {
                    return err("corrupt gzip-compressed tile: unexpected size");
                }
                if t.algo == Algo::Gzip2 {
                    buf = unshuffle(&buf, n, int_size);
                }
                (0..n).map(|i| be_int(&buf[i * int_size..(i + 1) * int_size], int_size)).collect()
            }
        };
        if !is_quantized {
            return Ok(ints.into_iter().map(|v| t.bzero + t.bscale * v as f64).collect());
        }
        // floating-point image stored as quantized integers: undo the
        // quantization (and the dither, if any) with this tile's scale/zero.
        let scale = match &t.zscale {
            Some(c) => column_f64(t, mmap, c, tile_index)?,
            None => return err("the compressed-image table has no ZSCALE column"),
        };
        let zero = match &t.zzero {
            Some(c) => column_f64(t, mmap, c, tile_index)?,
            None => return err("the compressed-image table has no ZZERO column"),
        };
        let tnull = t.zblank_col.and_then(|c| column_i64(t, mmap, &c, tile_index)).or(t.zblank);
        let rand = random_table();
        let mut iseed = ((tile_index as i64 + t.dither_seed - 1).rem_euclid(N_RANDOM as i64)) as usize;
        let mut nextrand = (rand[iseed] * 500.0) as usize;
        let mut out = Vec::with_capacity(n);
        for &iv in &ints {
            let v = if Some(iv) == tnull {
                f64::NAN
            } else if t.dither == 2 && iv == ZERO_VALUE {
                0.0
            } else if t.dither == 0 {
                iv as f64 * scale + zero
            } else {
                (iv as f64 - rand[nextrand] as f64 + 0.5) * scale + zero
            };
            out.push(if t.zbitpix == -32 { v as f32 as f64 } else { v });
            if t.dither != 0 {
                nextrand += 1;
                if nextrand == N_RANDOM {
                    iseed += 1;
                    if iseed == N_RANDOM {
                        iseed = 0;
                    }
                    nextrand = (rand[iseed] * 500.0) as usize;
                }
            }
        }
        return Ok(out);
    }

    let size = (t.zbitpix.unsigned_abs() as usize) / 8;
    if let Some(bytes) = nonempty(&t.gzip)? {
        // lossless gzip of the raw samples (used for floats that were not quantized)
        let mut buf = gunzip(bytes, n * size)?;
        if t.algo == Algo::Gzip2 {
            buf = unshuffle(&buf, n, size);
        }
        return samples_to_f64(t, &buf, n);
    }
    if let Some(bytes) = nonempty(&t.uncomp)? {
        return samples_to_f64(t, bytes, n);
    }
    err(format!("tile {tile_index} has no data in any of the table's data columns"))
}

/// Raw values of one tile of a 64-bit integer image. Only gzip can compress
/// those (cfitsio has no Rice / HCOMPRESS / PLIO for 64 bits).
fn decode_tile_i64(t: &Tiled, mmap: &[u8], tile_index: usize) -> PyResult<Vec<i64>> {
    let ntx = t.naxis1.div_ceil(t.tile1);
    let (tx, ty) = (tile_index % ntx, tile_index / ntx);
    let n = t.tile1.min(t.naxis1 - tx * t.tile1) * t.tile2.min(t.naxis2 - ty * t.tile2);
    let nonempty = |c: &Option<Column>| -> PyResult<Option<&[u8]>> {
        match c {
            Some(col) => Ok(Some(heap_bytes(t, mmap, col, tile_index)?).filter(|b| !b.is_empty())),
            None => Ok(None),
        }
    };
    let parse = |buf: &[u8]| -> PyResult<Vec<i64>> {
        if buf.len() < n * 8 {
            return err("corrupt tile: fewer bytes than the tile size implies");
        }
        Ok(buf.chunks_exact(8).take(n).map(|b| i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])).collect())
    };
    for column in [&t.comp, &t.gzip] {
        if let Some(bytes) = nonempty(column)? {
            if t.algo == Algo::Rice {
                return err("a Rice-compressed 64-bit integer tile cannot be decoded (the format does not allow it)");
            }
            let mut buf = gunzip(bytes, n * 8)?;
            if t.algo == Algo::Gzip2 {
                buf = unshuffle(&buf, n, 8);
            }
            return parse(&buf);
        }
    }
    if let Some(bytes) = nonempty(&t.uncomp)? {
        return parse(bytes);
    }
    err(format!("tile {tile_index} has no data in any of the table's data columns"))
}

/// Finds the requested HDU, checks that it is a plain 2-D image (raw or
/// tile-compressed) and returns what is needed to read its pixels. With
/// `hdu=None` the primary HDU is used.
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
        let tiled = build_tiled(h, hdu.data_offset).map_err(|e| PyValueError::new_err(format!("{label}: {e}")))?;
        return Ok(FitsHeader {
            bitpix: tiled.zbitpix,
            naxis1: tiled.naxis1,
            naxis2: tiled.naxis2,
            bzero: h.bzero,
            bscale: h.bscale,
            data_offset: 0,
            wcs: build_wcs(h).0,
            wcs_error: build_wcs(h).1,
            tiled: Some(tiled),
        });
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
        wcs: build_wcs(h).0,
        wcs_error: build_wcs(h).1,
        tiled: None,
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
    readable: bool, // True if euclid_fast can read it (2-D image, supported BITPIX / compression)
    #[pyo3(get)]
    note: String, // e.g. "tile-compressed RICE_1", or why it is not readable
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
            if is_compressed_image(h) {
                return match build_tiled(h, x.data_offset) {
                    Ok(t) => HduInfo {
                        index: x.index,
                        kind: hdu_kind(h),
                        name: h.extname.clone().unwrap_or_default(),
                        bitpix: Some(t.zbitpix),
                        shape: vec![t.naxis1, t.naxis2],
                        readable: true,
                        note: format!("tile-compressed {}", t.algo.name()),
                    },
                    Err(e) => HduInfo {
                        index: x.index,
                        kind: hdu_kind(h),
                        name: h.extname.clone().unwrap_or_default(),
                        bitpix: h.zbitpix,
                        shape: h.znaxes.clone(),
                        readable: false,
                        note: e,
                    },
                };
            }
            let naxis = h.naxis.unwrap_or(0);
            let supported_bitpix = matches!(h.bitpix, Some(8) | Some(16) | Some(32) | Some(64) | Some(-32) | Some(-64));
            let (readable, note) = if !is_image_hdu(h) {
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
    min_int: Option<i128>, // exact min / max for 64-bit integer images, else None
    #[pyo3(get)]
    max_int: Option<i128>,
    #[pyo3(get)]
    threads_used: usize,
    #[pyo3(get)]
    elapsed_ms: f64,
}

/// (min, max, sum) of a slice of physical values.
fn reduce_f64(v: &[f64]) -> (f64, f64, f64) {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    let mut sum = 0.0_f64;
    for &x in v {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
        sum += x;
    }
    (mn, mx, sum)
}

/// Computes min/max/mean of one image HDU. Pure Rust (no Python objects), so it
/// can run with the GIL released and from several threads at once.
fn stats_impl(path: &str, hdu: &Option<HduSelector>) -> PyResult<FitsStats> {
    let start = std::time::Instant::now();

    let mmap = open_mmap(path)?;
    let header = locate_image(&mmap, hdu)?;
    let n_pixels = header.naxis1 * header.naxis2;

    let mut exact: Option<(i128, i128)> = None;
    let (min, max, sum) = if let Some(t) = &header.tiled {
        let affine = if t.zbitpix == 64 { int64_affine(header.bzero, header.bscale) } else { None };
        if let Some(affine) = affine {
            // 64-bit integer image (only gzip can compress those): exact arithmetic
            let parts: Vec<(i64, i64, i128)> = (0..t.nrows)
                .into_par_iter()
                .map(|i| decode_tile_i64(t, &mmap, i).map(|v| reduce_i64_slice(&v)))
                .collect::<PyResult<Vec<_>>>()?;
            let raw = parts.into_iter().fold((i64::MAX, i64::MIN, 0i128), |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 + b.2));
            let (mn, mx, sm) = affine_stats(raw, n_pixels as i128, affine);
            exact = Some((mn, mx));
            (mn as f64, mx as f64, sm as f64)
        } else {
            let parts: Vec<(f64, f64, f64)> = (0..t.nrows)
                .into_par_iter()
                .map(|i| decode_tile(t, &mmap, i).map(|v| reduce_f64(&v)))
                .collect::<PyResult<Vec<_>>>()?;
            parts.into_iter().fold((f64::INFINITY, f64::NEG_INFINITY, 0.0), |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 + b.2))
        }
    } else {
        let bytes_per_pixel = (header.bitpix.unsigned_abs() as usize) / 8;
        let data_end = header.data_offset + n_pixels * bytes_per_pixel;
        if data_end > mmap.len() {
            return err("truncated file: fewer pixels than the header announces");
        }
        let data = &mmap[header.data_offset..data_end];

        // Read + convert to "physical" f64 (BZERO + BSCALE * raw_value), in
        // parallel over chunks of pixels. rayon sizes the number of threads
        // itself from the machine's logical core count.
        match header.bitpix {
            16 => reduce_be::<i16>(data, header.bzero, header.bscale),
            8 => reduce_be::<u8>(data, header.bzero, header.bscale),
            32 => reduce_be::<i32>(data, header.bzero, header.bscale),
            64 => match int64_affine(header.bzero, header.bscale) {
                // Exact integer arithmetic: BZERO and BSCALE are applied as integers, so
                // unsigned 64-bit data (BZERO = 2^63) and values above 2^53 stay exact.
                Some(affine) => {
                    let (mn, mx, sm) = affine_stats(reduce_i64_raw(data), (data.len() / 8) as i128, affine);
                    exact = Some((mn, mx));
                    (mn as f64, mx as f64, sm as f64)
                }
                None => reduce_be::<i64>(data, header.bzero, header.bscale),
            },
            -32 => reduce_be::<f32>(data, header.bzero, header.bscale),
            -64 => reduce_be::<f64>(data, header.bzero, header.bscale),
            other => return err(format!("BITPIX {other} is not supported yet")),
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
        min_int: exact.map(|e| e.0),
        max_int: exact.map(|e| e.1),
        threads_used: rayon::current_num_threads(),
        elapsed_ms,
    })
}


/// Reads a FITS image (2-D, raw or tile-compressed) and computes min/max/mean
/// in parallel with rayon. Raw images are read through mmap (zero-copy for the
/// disk -> memory part); compressed images are decoded tile by tile, the tiles
/// being spread over the threads. The rayon thread pool sizes itself to the
/// number of cores available on the machine (no hardcoded value). The GIL is
/// released while Rust works, so other Python threads keep running. `hdu`
/// selects the HDU by 0-based index or EXTNAME; by default the primary HDU is
/// read.
#[pyfunction]
#[pyo3(signature = (path, hdu=None))]
fn fits_stats(python: Python<'_>, path: &str, hdu: Option<HduSelector>) -> PyResult<FitsStats> {
    python.detach(|| stats_impl(path, &hdu))
}

/// Statistics of one HDU, as returned by `fits_stats_all_hdus`.
#[pyclass]
struct HduStats {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    name: String,
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
    min_int: Option<i128>,
    #[pyo3(get)]
    max_int: Option<i128>,
    #[pyo3(get)]
    elapsed_ms: f64,
}

/// Prefixes an error with the file (or position) it comes from.
fn tag_error(python: Python<'_>, what: &str, e: PyErr) -> PyErr {
    PyValueError::new_err(format!("{what}: {}", e.value(python)))
}

/// Statistics of many files at once. The files are processed in parallel on
/// all the cores, with the GIL released. `hdu` applies to every file. With
/// `skip_errors=True`, a file that cannot be read gives `None` in the result
/// instead of raising an error.
#[pyfunction]
#[pyo3(signature = (paths, hdu=None, skip_errors=false))]
fn fits_stats_many(
    python: Python<'_>,
    paths: Vec<String>,
    hdu: Option<HduSelector>,
    skip_errors: bool,
) -> PyResult<Vec<Option<FitsStats>>> {
    let results: Vec<PyResult<FitsStats>> =
        python.detach(|| paths.par_iter().map(|p| stats_impl(p, &hdu)).collect());
    let mut out = Vec::with_capacity(results.len());
    for (path, r) in paths.iter().zip(results) {
        match r {
            Ok(s) => out.push(Some(s)),
            Err(e) if skip_errors => {
                let _ = e;
                out.push(None)
            }
            Err(e) => return Err(tag_error(python, path, e)),
        }
    }
    Ok(out)
}

/// Whether an HDU is an image that `euclid_fast` can read.
fn is_readable(h: &RawHeader, data_offset: usize) -> bool {
    if is_compressed_image(h) {
        return build_tiled(h, data_offset).is_ok();
    }
    is_image_hdu(h)
        && h.naxis == Some(2)
        && matches!(h.bitpix, Some(8) | Some(16) | Some(32) | Some(64) | Some(-32) | Some(-64))
}

/// Statistics of every readable image HDU of one file (all the detectors of a
/// Euclid frame, for example), computed in parallel with the GIL released.
/// `name_contains` keeps only the HDUs whose EXTNAME contains that text
/// (case-insensitive), e.g. "SCI" to skip the RMS and DQ images.
#[pyfunction]
#[pyo3(signature = (path, name_contains=None))]
fn fits_stats_all_hdus(
    python: Python<'_>,
    path: &str,
    name_contains: Option<String>,
) -> PyResult<Vec<HduStats>> {
    python.detach(|| {
        let mmap = open_mmap(path)?;
        let needle = name_contains.map(|s| s.to_ascii_lowercase());
        let wanted: Vec<(usize, String)> = all_hdus(&mmap)?
            .iter()
            .filter(|x| is_readable(&x.header, x.data_offset))
            .map(|x| (x.index, x.header.extname.clone().unwrap_or_default()))
            .filter(|(_, name)| needle.as_ref().is_none_or(|n| name.to_ascii_lowercase().contains(n)))
            .collect();
        wanted
            .par_iter()
            .map(|(index, name)| {
                let sel = Some(HduSelector::Index(*index));
                stats_impl(path, &sel).map(|s| HduStats {
                    index: *index,
                    name: name.clone(),
                    naxis1: s.naxis1,
                    naxis2: s.naxis2,
                    bitpix: s.bitpix,
                    min: s.min,
                    max: s.max,
                    mean: s.mean,
                    min_int: s.min_int,
                    max_int: s.max_int,
                    elapsed_ms: s.elapsed_ms,
                })
            })
            .collect()
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

/// BZERO and BSCALE as whole numbers, when the physical value of a 64-bit integer
/// pixel, `BZERO + BSCALE * raw`, can then be computed exactly in integers. This
/// covers signed 64-bit images (BZERO = 0, BSCALE = 1) and unsigned ones, which
/// FITS stores as signed with BZERO = 2^63. The bounds keep every intermediate
/// result (including sums over 2^40 pixels) inside 128 bits.
fn int64_affine(bzero: f64, bscale: f64) -> Option<(i128, i128)> {
    let whole = |x: f64| x.fract() == 0.0;
    if whole(bzero) && whole(bscale) && bscale != 0.0 && bzero.abs() <= 18446744073709551616.0 && bscale.abs() <= 1048576.0 {
        Some((bzero as i128, bscale as i128))
    } else {
        None
    }
}

/// (min, max, sum) of a slice of raw 64-bit integers; the sum is exact.
fn reduce_i64_slice(v: &[i64]) -> (i64, i64, i128) {
    let (mut mn, mut mx, mut sum) = (i64::MAX, i64::MIN, 0i128);
    for &x in v {
        mn = mn.min(x);
        mx = mx.max(x);
        sum += x as i128;
    }
    (mn, mx, sum)
}

/// Raw (min, max, sum) of big-endian 64-bit integers, in parallel.
fn reduce_i64_raw(data: &[u8]) -> (i64, i64, i128) {
    data.par_chunks(8 << 16)
        .map(|chunk| {
            let (mut mn, mut mx, mut sum) = (i64::MAX, i64::MIN, 0i128);
            for b in chunk.chunks_exact(8) {
                let v = i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                mn = mn.min(v);
                mx = mx.max(v);
                sum += v as i128;
            }
            (mn, mx, sum)
        })
        .reduce(|| (i64::MAX, i64::MIN, 0i128), |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 + b.2))
}

/// Exact physical (min, max, sum) from the raw reduction of `n` pixels:
/// each value is `bzero + bscale * raw` (a negative BSCALE swaps min and max).
fn affine_stats(raw: (i64, i64, i128), n: i128, (bzero, bscale): (i128, i128)) -> (i128, i128, i128) {
    let (a, b) = (bzero + bscale * raw.0 as i128, bzero + bscale * raw.1 as i128);
    (a.min(b), a.max(b), bzero * n + bscale * raw.2)
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
    let residual = |x: f64, y: f64| -> (f64, f64) {
        let (fx, fy) = tpv_forward(t, x, y);
        (fx - xi, fy - eta)
    };
    let (mut x, mut y) = (xi, eta);
    let h = 1e-6; // finite-difference step, degrees
    for _ in 0..80 {
        let (rx, ry) = residual(x, y);
        if rx.abs() < 1e-12 && ry.abs() < 1e-12 {
            return Ok((x, y));
        }
        let (fx, fy) = (rx + xi, ry + eta);
        let (fx_dx, fy_dx) = tpv_forward(t, x + h, y);
        let (fx_dy, fy_dy) = tpv_forward(t, x, y + h);
        let (a, c) = ((fx_dx - fx) / h, (fy_dx - fy) / h); // d(fx,fy)/dx
        let (b, d) = ((fx_dy - fx) / h, (fy_dy - fy) / h); // d(fx,fy)/dy
        let det = a * d - b * c;
        if det == 0.0 || !det.is_finite() {
            return err("TPV distortion cannot be inverted at this position (singular Jacobian)");
        }
        let (dx, dy) = ((d * rx - b * ry) / det, (-c * rx + a * ry) / det);
        // Damped Newton: halve the step until the residual shrinks.
        let norm0 = rx.hypot(ry);
        let mut lambda = 1.0;
        for _ in 0..40 {
            let (r2x, r2y) = residual(x - lambda * dx, y - lambda * dy);
            if r2x.hypot(r2y) < norm0 {
                break;
            }
            lambda *= 0.5;
        }
        x -= lambda * dx;
        y -= lambda * dy;
    }
    err("TPV distortion inversion did not converge")
}

/// Sky position -> intermediate world coordinates (x, y) in degrees, for a
/// zenithal projection (Calabretta & Greisen 2002): rotate to native spherical
/// coordinates (phi, theta) around the reference point, then apply the
/// projection's R(theta).
fn zenithal_xy(wcs: &Wcs, ra_deg: f64, dec_deg: f64) -> PyResult<(f64, f64)> {
    let a = ra_deg.to_radians();
    let d = dec_deg.to_radians();
    let da = a - wcs.crval1.to_radians();
    let (sd, cd_) = d.sin_cos();
    let (sd0, cd0) = wcs.crval2.to_radians().sin_cos();

    let sin_theta = sd * sd0 + cd_ * cd0 * da.cos();
    let u = -cd_ * da.sin();
    let w = sd * cd0 - cd_ * sd0 * da.cos();
    let cos_theta = u.hypot(w);
    let phi = wcs.lonpole.to_radians() + u.atan2(w);
    let zeta = cos_theta.atan2(sin_theta); // 90 degrees - theta, in radians

    let r = match wcs.proj {
        Proj::Tan | Proj::Sin if sin_theta <= 1e-9 => {
            return err("the requested sky position is not within 90 degrees of the image center (no projection)")
        }
        Proj::Tan => cos_theta / sin_theta,
        Proj::Sin => cos_theta,
        Proj::Arc => zeta,
        Proj::Stg => {
            if zeta > std::f64::consts::PI - 1e-9 {
                return err("the requested sky position is opposite the image center (no stereographic projection)");
            }
            2.0 * (zeta / 2.0).tan()
        }
        Proj::Zea => 2.0 * (zeta / 2.0).sin(),
        Proj::Zpn => wcs.zpn[..31].iter().rev().fold(0.0, |acc, &c| acc * zeta + c),
    };
    let r = r.to_degrees();
    Ok((r * phi.sin(), -r * phi.cos()))
}

/// Inverts the SIP distortion with Newton's method: finds the pixel offsets
/// (u, v) such that u + f(u, v) = U and v + g(u, v) = V.
fn sip_invert(sip: &Sip, big_u: f64, big_v: f64) -> PyResult<(f64, f64)> {
    let poly = |terms: &[(i32, i32, f64)], u: f64, v: f64| -> (f64, f64, f64) {
        // value and the two partial derivatives
        let (mut s, mut su, mut sv) = (0.0, 0.0, 0.0);
        for &(p, q, c) in terms {
            s += c * u.powi(p) * v.powi(q);
            if p > 0 {
                su += c * p as f64 * u.powi(p - 1) * v.powi(q);
            }
            if q > 0 {
                sv += c * q as f64 * u.powi(p) * v.powi(q - 1);
            }
        }
        (s, su, sv)
    };
    let residual = |u: f64, v: f64| -> (f64, f64) {
        (u + poly(&sip.a, u, v).0 - big_u, v + poly(&sip.b, u, v).0 - big_v)
    };
    // First, a fixed-point iteration u = U - f(u, v): for the mild distortions of
    // real instruments it is a contraction that converges to the root nearest
    // to the undistorted position, which is the one we want even if the
    // distortion polynomial folds over far from the image.
    let (mut u, mut v) = (big_u, big_v);
    for _ in 0..200 {
        let (ru, rv) = residual(u, v);
        if ru.abs() < 1e-11 && rv.abs() < 1e-11 {
            return Ok((u, v));
        }
        let (nu, nv) = (big_u - poly(&sip.a, u, v).0, big_v - poly(&sip.b, u, v).0);
        if !nu.is_finite() || !nv.is_finite() || nu.abs() > 1e7 || nv.abs() > 1e7 {
            break; // diverging: fall back to Newton's method
        }
        u = nu;
        v = nv;
    }
    // Fallback: damped Newton from the undistorted position.
    let (mut u, mut v) = (big_u, big_v);
    for _ in 0..80 {
        let (ru, rv) = residual(u, v);
        if ru.abs() < 1e-11 && rv.abs() < 1e-11 {
            return Ok((u, v));
        }
        let (_, fa_u, fa_v) = poly(&sip.a, u, v);
        let (_, fb_u, fb_v) = poly(&sip.b, u, v);
        let (j11, j12, j21, j22) = (1.0 + fa_u, fa_v, fb_u, 1.0 + fb_v);
        let det = j11 * j22 - j12 * j21;
        if det == 0.0 || !det.is_finite() {
            return err("SIP distortion cannot be inverted at this position (singular Jacobian)");
        }
        let (du, dv) = ((j22 * ru - j12 * rv) / det, (-j21 * ru + j11 * rv) / det);
        // Damped Newton: halve the step until the residual shrinks, so the
        // iteration stays on the root nearest to the starting guess.
        let norm0 = ru.hypot(rv);
        let mut lambda = 1.0;
        for _ in 0..40 {
            let (nu, nv) = (u - lambda * du, v - lambda * dv);
            let (r2u, r2v) = residual(nu, nv);
            if r2u.hypot(r2v) < norm0 {
                break;
            }
            lambda *= 0.5;
        }
        u -= lambda * du;
        v -= lambda * dv;
    }
    err("SIP distortion inversion did not converge")
}

/// Converts sky coordinates (RA/Dec, in degrees) to pixel coordinates (FITS
/// 1-indexed convention): zenithal projection (TAN, SIN, ARC, STG, ZEA, ZPN),
/// then the TPV polynomial if the header has one, the inverse of the CD matrix
/// and finally the SIP distortion if the header has one.
fn radec_to_pixel(ra_deg: f64, dec_deg: f64, wcs: &Wcs) -> PyResult<(f64, f64)> {
    let (xi, eta) = zenithal_xy(wcs, ra_deg, dec_deg)?;

    // With TPV, the tangent-plane coordinates above are the *distorted* ones:
    // undo the polynomial to get the intermediate world coordinates.
    let (xw, yw) = match &wcs.tpv {
        Some(t) => tpv_invert(t, xi, eta)?,
        None => (xi, eta),
    };

    // Invert the 2x2 matrix to go from intermediate world coordinates to
    // pixel offsets.
    let [cd11, cd12, cd21, cd22] = wcs.cd;
    let det = cd11 * cd22 - cd12 * cd21;
    if det == 0.0 || !det.is_finite() {
        return err("the WCS matrix is singular");
    }
    let big_u = (cd22 * xw - cd12 * yw) / det;
    let big_v = (-cd21 * xw + cd11 * yw) / det;

    // SIP distortion applies to the pixel offsets, before the matrix.
    let (u, v) = match &wcs.sip {
        Some(s) => sip_invert(s, big_u, big_v)?,
        None => (big_u, big_v),
    };
    Ok((wcs.crpix1 + u, wcs.crpix2 + v))
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
    data_int: Option<Vec<i128>>, // exact values for 64-bit integer images, else None
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
) -> PyResult<(f64, f64, usize, usize, Vec<f64>, Option<Vec<i128>>)> {
    let mmap = open_mmap(path)?;
    let header = locate_image(&mmap, hdu)?;
    cutout_at(&mmap, &header, ra_deg, dec_deg, width, height)
}

/// One cutout from an already opened file and located HDU.
fn cutout_at(
    mmap: &[u8],
    header: &FitsHeader,
    ra_deg: f64,
    dec_deg: f64,
    width: usize,
    height: usize,
) -> PyResult<(f64, f64, usize, usize, Vec<f64>, Option<Vec<i128>>)> {
    if let Some(msg) = &header.wcs_error {
        return err(msg.clone());
    }
    let wcs = header.wcs.as_ref().ok_or_else(|| {
        PyValueError::new_err("this file has no WCS header (CRVAL, CRPIX and either a CD matrix or CDELT/PC)")
    })?;

    let (px, py) = radec_to_pixel(ra_deg, dec_deg, wcs)?;
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

    if let Some(t) = &header.tiled {
        // Compressed image: decode only the tiles that overlap the cutout.
        let ntx = t.naxis1.div_ceil(t.tile1);
        let (tx_lo, tx_hi) = (x0 / t.tile1, (x0 + width - 1) / t.tile1);
        let (ty_lo, ty_hi) = (y0 / t.tile2, (y0 + height - 1) / t.tile2);
        let wanted: Vec<(usize, usize)> =
            (ty_lo..=ty_hi).flat_map(|ty| (tx_lo..=tx_hi).map(move |tx| (tx, ty))).collect();
        let affine = if t.zbitpix == 64 { int64_affine(header.bzero, header.bscale) } else { None };
        if let Some((bz, bs)) = affine {
            // 64-bit integer image: keep the exact integers
            let decoded: Vec<Vec<i64>> = wanted
                .par_iter()
                .map(|&(tx, ty)| decode_tile_i64(t, mmap, ty * ntx + tx))
                .collect::<PyResult<Vec<_>>>()?;
            let mut ints = vec![0i128; width * height];
            for (&(tx, ty), tile) in wanted.iter().zip(decoded.iter()) {
                let (tile_x0, tile_y0) = (tx * t.tile1, ty * t.tile2);
                let tile_w = t.tile1.min(t.naxis1 - tile_x0);
                for row in 0..height {
                    let y = y0 + row;
                    if y < tile_y0 || y >= tile_y0 + t.tile2.min(t.naxis2 - tile_y0) {
                        continue;
                    }
                    for col in 0..width {
                        let x = x0 + col;
                        if x < tile_x0 || x >= tile_x0 + tile_w {
                            continue;
                        }
                        ints[row * width + col] = bz + bs * tile[(y - tile_y0) * tile_w + (x - tile_x0)] as i128;
                    }
                }
            }
            let data: Vec<f64> = ints.iter().map(|&v| v as f64).collect();
            return Ok((px, py, x0, y0, data, Some(ints)));
        }
        let decoded: Vec<Vec<f64>> = wanted
            .par_iter()
            .map(|&(tx, ty)| decode_tile(t, mmap, ty * ntx + tx))
            .collect::<PyResult<Vec<_>>>()?;

        let mut data = vec![0.0f64; width * height];
        for (&(tx, ty), tile) in wanted.iter().zip(decoded.iter()) {
            let (tile_x0, tile_y0) = (tx * t.tile1, ty * t.tile2);
            let tile_w = t.tile1.min(t.naxis1 - tile_x0);
            for row in 0..height {
                let y = y0 + row;
                if y < tile_y0 || y >= tile_y0 + t.tile2.min(t.naxis2 - tile_y0) {
                    continue;
                }
                for col in 0..width {
                    let x = x0 + col;
                    if x < tile_x0 || x >= tile_x0 + tile_w {
                        continue;
                    }
                    data[row * width + col] = tile[(y - tile_y0) * tile_w + (x - tile_x0)];
                }
            }
        }
        return Ok((px, py, x0, y0, data, None));
    }

    let bytes_per_pixel = (header.bitpix.unsigned_abs() as usize) / 8;
    let row_stride = header.naxis1 * bytes_per_pixel;
    let last_byte = header.data_offset + (y0 + height) * row_stride;
    if last_byte > mmap.len() {
        return err("truncated file: fewer pixels than the header announces");
    }

    let mut data = Vec::with_capacity(width * height);
    // 64-bit integers are read exactly (BZERO = 2^63 is how FITS stores uint64).
    let exact_affine = if header.bitpix == 64 { int64_affine(header.bzero, header.bscale) } else { None };
    let mut data_int: Option<Vec<i128>> = exact_affine.map(|_| Vec::with_capacity(width * height));
    for row in 0..height {
        let row_offset = header.data_offset + (y0 + row) * row_stride + x0 * bytes_per_pixel;
        let row_bytes = &mmap[row_offset..row_offset + width * bytes_per_pixel];
        for col in 0..width {
            let off = col * bytes_per_pixel;
            let raw = &row_bytes[off..off + bytes_per_pixel];
            if let (Some((bz, bs)), Some(ints)) = (exact_affine, data_int.as_mut()) {
                let value = bz + bs * i64::from_be_bytes([raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]]) as i128;
                ints.push(value);
                data.push(value as f64);
                continue;
            }
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

    Ok((px, py, x0, y0, data, data_int))
}

/// Extracts a cutout (small rectangle) around sky coordinates (RA/Dec in
/// degrees) without reading the whole file: through mmap only the few
/// rows needed are touched -- the OS only loads into RAM the memory pages
/// that correspond to the bytes actually read (that is the project's whole
/// point: no need to read 1 GB to extract a thumbnail). For tile-compressed
/// images only the tiles overlapping the cutout are decompressed. `hdu`
/// selects the HDU by 0-based index or EXTNAME; by default the primary HDU
/// is read.
#[pyfunction]
#[pyo3(signature = (path, ra_deg, dec_deg, width, height, hdu=None))]
fn extract_cutout(
    python: Python<'_>,
    path: &str,
    ra_deg: f64,
    dec_deg: f64,
    width: usize,
    height: usize,
    hdu: Option<HduSelector>,
) -> PyResult<Cutout> {
    let start = std::time::Instant::now();
    let (px, py, _x0, _y0, data, data_int) =
        python.detach(|| extract_cutout_raw(path, ra_deg, dec_deg, width, height, &hdu))?;
    Ok(Cutout {
        width,
        height,
        pixel_x_center: px,
        pixel_y_center: py,
        data,
        data_int,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Many cutouts from one file: the file is opened and its header read once,
/// then the positions are processed in parallel on all the cores, with the GIL
/// released. `ra_deg` and `dec_deg` are two equal-length lists (or numpy
/// arrays). With `skip_errors=True`, a position that cannot be extracted (for
/// example outside the image) gives `None` instead of raising an error.
#[pyfunction]
#[pyo3(signature = (path, ra_deg, dec_deg, width, height, hdu=None, skip_errors=false))]
fn extract_cutouts(
    python: Python<'_>,
    path: &str,
    ra_deg: Vec<f64>,
    dec_deg: Vec<f64>,
    width: usize,
    height: usize,
    hdu: Option<HduSelector>,
    skip_errors: bool,
) -> PyResult<Vec<Option<Cutout>>> {
    if ra_deg.len() != dec_deg.len() {
        return err(format!(
            "ra_deg and dec_deg must have the same length ({} and {})",
            ra_deg.len(),
            dec_deg.len()
        ));
    }
    let results: Vec<PyResult<Cutout>> = python.detach(|| -> PyResult<Vec<PyResult<Cutout>>> {
        let mmap = open_mmap(path)?;
        let header = locate_image(&mmap, &hdu)?;
        Ok(ra_deg
            .par_iter()
            .zip(dec_deg.par_iter())
            .map(|(&ra, &dec)| {
                let start = std::time::Instant::now();
                cutout_at(&mmap, &header, ra, dec, width, height).map(|(px, py, _x0, _y0, data, data_int)| Cutout {
                    width,
                    height,
                    pixel_x_center: px,
                    pixel_y_center: py,
                    data,
                    data_int,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                })
            })
            .collect())
    })?;
    let mut out = Vec::with_capacity(results.len());
    for (i, r) in results.into_iter().enumerate() {
        match r {
            Ok(c) => out.push(Some(c)),
            Err(e) if skip_errors => {
                let _ = e;
                out.push(None)
            }
            Err(e) => {
                return Err(tag_error(python, &format!("position {i} (ra={}, dec={})", ra_deg[i], dec_deg[i]), e))
            }
        }
    }
    Ok(out)
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
    python: Python<'_>,
    path: &str,
    ra_deg: f64,
    dec_deg: f64,
    width: usize,
    height: usize,
    out_path: &str,
    hdu: Option<HduSelector>,
) -> PyResult<ParquetExport> {
    let start = std::time::Instant::now();
    let (_px, _py, x0, y0, data, data_int) =
        python.detach(|| extract_cutout_raw(path, ra_deg, dec_deg, width, height, &hdu))?;

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
    // 64-bit integer images keep their exact values: an Int64 column when they fit
    // (signed data), else UInt64 (unsigned data); anything else is Float64.
    let v_arr: arrow_array::ArrayRef = match &data_int {
        Some(ints) if ints.iter().all(|&v| v >= i64::MIN as i128 && v <= i64::MAX as i128) => {
            std::sync::Arc::new(arrow_array::Int64Array::from(ints.iter().map(|&v| v as i64).collect::<Vec<_>>()))
        }
        Some(ints) if ints.iter().all(|&v| v >= 0 && v <= u64::MAX as i128) => {
            std::sync::Arc::new(arrow_array::UInt64Array::from(ints.iter().map(|&v| v as u64).collect::<Vec<_>>()))
        }
        _ => std::sync::Arc::new(arrow_array::Float64Array::from(data)),
    };

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
    m.add_function(wrap_pyfunction!(fits_stats_many, m)?)?;
    m.add_function(wrap_pyfunction!(fits_stats_all_hdus, m)?)?;
    m.add_function(wrap_pyfunction!(extract_cutouts, m)?)?;
    m.add_function(wrap_pyfunction!(detected_cpus, m)?)?;
    m.add_class::<FitsStats>()?;
    m.add_class::<Cutout>()?;
    m.add_class::<ParquetExport>()?;
    m.add_class::<HduInfo>()?;
    m.add_class::<HduStats>()?;
    Ok(())
}
