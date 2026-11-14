use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;

const CARD_LEN: usize = 80;
const BLOCK_LEN: usize = 2880; // 36 cards de 80 octets

/// Paramètres WCS (World Coordinate System) : la transformation standard
/// FITS "pixel <-> coordonnées célestes" en projection tangentielle (TAN).
/// Absents si le fichier n'a pas d'astrométrie moderne (ex: horsehead.fits
/// qui utilise une vieille astrométrie de plaque non standard).
#[derive(Debug, Clone, Copy)]
struct Wcs {
    crval1: f64, // RA du pixel de référence, en degrés
    crval2: f64, // Dec du pixel de référence, en degrés
    crpix1: f64, // pixel de référence, axe 1 (convention FITS, 1-indexé)
    crpix2: f64, // pixel de référence, axe 2
    cd1_1: f64,  // matrice WCS, degrés/pixel
    cd1_2: f64,
    cd2_1: f64,
    cd2_2: f64,
}

/// En-tête FITS minimal dont on a besoin pour lire la matrice de pixels.
#[derive(Debug, Clone, Copy)]
struct FitsHeader {
    bitpix: i32,
    naxis1: usize,
    naxis2: usize,
    bzero: f64,
    bscale: f64,
    data_offset: usize, // position en octets où commencent les pixels
    wcs: Option<Wcs>,
}

/// Parse les cartes d'en-tête FITS (blocs de 2880 octets = 36 cartes de 80
/// octets) jusqu'à la carte "END". Ne gère que les images simples (1 HDU,
/// NAXIS=2, pas d'extensions) : suffisant pour notre jeu de données de dev.
fn parse_header(mmap: &[u8]) -> PyResult<FitsHeader> {
    let mut bitpix: Option<i32> = None;
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
                "fichier tronqué : pas de carte END trouvée",
            ));
        }
        let block = &mmap[block_start..block_start + BLOCK_LEN];
        for card_idx in 0..(BLOCK_LEN / CARD_LEN) {
            let card = &block[card_idx * CARD_LEN..(card_idx + 1) * CARD_LEN];
            let card_str = std::str::from_utf8(card).unwrap_or("");
            let keyword = card_str[0..8].trim();

            if keyword == "END" {
                let data_offset = block_start + BLOCK_LEN;
                let bitpix = bitpix.ok_or_else(|| PyValueError::new_err("BITPIX manquant"))?;
                let naxis1 = naxis1.ok_or_else(|| PyValueError::new_err("NAXIS1 manquant"))?;
                let naxis2 = naxis2.ok_or_else(|| PyValueError::new_err("NAXIS2 manquant"))?;
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
                });
            }

            if card_str.len() >= 10 && &card_str[8..10] == "= " {
                let value_part = &card_str[10..];
                let value_str = value_part.split('/').next().unwrap_or("").trim();
                match keyword {
                    "BITPIX" => bitpix = value_str.parse::<i32>().ok(),
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
            // Garde-fou : un en-tête FITS raisonnable ne dépasse pas ça.
            break 'blocks;
        }
    }
    Err(PyValueError::new_err("carte END introuvable"))
}

/// Statistiques calculées sur l'image : min, max, moyenne (valeurs physiques
/// après application de BZERO/BSCALE), + infos de contexte sur le calcul.
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

/// Lit un fichier FITS (image simple, non compressée) via mmap (zéro-copie
/// pour la partie lecture disque -> mémoire) et calcule min/max/mean en
/// parallèle avec rayon. Le pool de threads rayon s'adapte automatiquement
/// au nombre de coeurs disponibles sur la machine (pas de valeur en dur).
#[pyfunction]
fn fits_stats(path: &str) -> PyResult<FitsStats> {
    let start = std::time::Instant::now();

    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("impossible d'ouvrir {path}: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| PyValueError::new_err(format!("mmap échoué: {e}")))?
    };

    let header = parse_header(&mmap)?;
    let n_pixels = header.naxis1 * header.naxis2;
    let bytes_per_pixel = (header.bitpix.unsigned_abs() as usize) / 8;
    let data_end = header.data_offset + n_pixels * bytes_per_pixel;

    if data_end > mmap.len() {
        return Err(PyValueError::new_err(
            "fichier tronqué : moins de pixels que ce que l'en-tête annonce",
        ));
    }
    let data = &mmap[header.data_offset..data_end];

    // Lecture + conversion en f64 "physique" (BZERO + BSCALE*valeur_brute),
    // en parallèle sur des chunks de pixels. rayon dimensionne lui-même le
    // nombre de threads au nombre de coeurs logiques de la machine.
    let (min, max, sum) = match header.bitpix {
        16 => reduce_be::<i16>(data, header.bzero, header.bscale),
        8 => reduce_be::<u8>(data, header.bzero, header.bscale),
        32 => reduce_be::<i32>(data, header.bzero, header.bscale),
        -32 => reduce_be::<f32>(data, header.bzero, header.bscale),
        -64 => reduce_be::<f64>(data, header.bzero, header.bscale),
        other => {
            return Err(PyValueError::new_err(format!(
                "BITPIX {other} non supporté pour l'instant"
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

/// Trait minimal pour convertir un échantillon big-endian brut en f64.
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

/// Réduction parallèle (min, max, somme) sur les pixels bruts d'un type T,
/// en appliquant BZERO/BSCALE pour obtenir la valeur physique.
///
/// v2 : au lieu de confier à rayon un itérateur pixel par pixel (overhead de
/// scheduling/fold par élément), on découpe les données en gros blocs
/// (`par_chunks`) traités chacun par une boucle séquentielle serrée — ce
/// style de boucle est ce que LLVM auto-vectorise le mieux (SIMD implicite,
/// sans intrinsics explicites ni nightly), tout en gardant le parallélisme
/// entre blocs via rayon.
fn reduce_be<T: BigEndianSample + Sync>(data: &[u8], bzero: f64, bscale: f64) -> (f64, f64, f64) {
    const CHUNK_PIXELS: usize = 1 << 16; // 65536 pixels par bloc rayon
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

/// Convertit des coordonnées célestes (RA/Dec, en degrés) en coordonnées
/// pixel (convention FITS 1-indexée), via projection tangentielle (TAN /
/// gnomonique) standard — la projection utilisée par la quasi-totalité des
/// relevés modernes (dont Euclid) pour les images à champ pas trop large.
/// Ne gère pas de distorsion SIP (termes d'ordre supérieur) : suffisant
/// pour une première version, à affiner si les vraies images Euclid en ont.
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

    // Inversion de la matrice CD 2x2 pour passer des coordonnées monde
    // intermédiaires (xi, eta) aux décalages en pixels.
    let det = wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1;
    let dx = (wcs.cd2_2 * xi_deg - wcs.cd1_2 * eta_deg) / det;
    let dy = (-wcs.cd2_1 * xi_deg + wcs.cd1_1 * eta_deg) / det;

    (wcs.crpix1 + dx, wcs.crpix2 + dy)
}

/// Résultat d'une extraction de cutout : la sous-image (valeurs physiques,
/// ligne par ligne) + le contexte (où elle a été prise, combien de temps).
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
    data: Vec<f64>, // row-major, longueur = width*height
    #[pyo3(get)]
    elapsed_ms: f64,
}

/// Coeur partagé de l'extraction de cutout (sans la mesure de temps ni le
/// habillage PyO3), réutilisé par `extract_cutout` et `export_cutout_parquet`.
/// Retourne (pixel_x_centre, pixel_y_centre, x0, y0, data row-major).
fn extract_cutout_raw(
    path: &str,
    ra_deg: f64,
    dec_deg: f64,
    width: usize,
    height: usize,
) -> PyResult<(f64, f64, usize, usize, Vec<f64>)> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("impossible d'ouvrir {path}: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| PyValueError::new_err(format!("mmap échoué: {e}")))?
    };
    let header = parse_header(&mmap)?;
    let wcs = header
        .wcs
        .ok_or_else(|| PyValueError::new_err("ce fichier n'a pas d'en-tête WCS (CRVAL/CRPIX/CD)"))?;

    let (px, py) = radec_to_pixel(ra_deg, dec_deg, &wcs);
    // Conversion 1-indexé FITS -> 0-indexé tableau, puis coin haut-gauche du cutout.
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
            "cutout hors image : pixel central=({px:.1},{py:.1}), image={}x{}",
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
                other => return Err(PyValueError::new_err(format!("BITPIX {other} non supporté"))),
            };
            data.push(header.bzero + header.bscale * v);
        }
    }

    Ok((px, py, x0, y0, data))
}

/// Extrait un cutout (petit carré) autour de coordonnées célestes (RA/Dec en
/// degrés), sans lire le fichier entier : via mmap, on ne touche que les
/// quelques lignes nécessaires — l'OS ne charge en RAM que les pages
/// mémoire correspondant aux octets réellement lus (c'est le gain promis
/// par le projet : pas besoin de lire 1 Go pour extraire une vignette).
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

/// Résultat d'un export Parquet.
#[pyclass]
struct ParquetExport {
    #[pyo3(get)]
    out_path: String,
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    elapsed_ms: f64,
}

/// Extrait un cutout ET l'exporte directement en Parquet (format colonnaire
/// moderne, lu instantanément par pandas/polars/DuckDB) : une ligne par
/// pixel, colonnes (x, y, value), x/y étant les coordonnées pixel absolues
/// dans l'image source (pas juste dans le cutout), pour pouvoir recoller
/// plusieurs cutouts ensemble si besoin.
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
        .map_err(|e| PyValueError::new_err(format!("construction du RecordBatch échouée: {e}")))?;

    let out_file = File::create(out_path)
        .map_err(|e| PyValueError::new_err(format!("création de {out_path} échouée: {e}")))?;
    let mut writer = parquet::arrow::arrow_writer::ArrowWriter::try_new(out_file, batch.schema(), None)
        .map_err(|e| PyValueError::new_err(format!("init ArrowWriter échouée: {e}")))?;
    writer
        .write(&batch)
        .map_err(|e| PyValueError::new_err(format!("écriture Parquet échouée: {e}")))?;
    writer
        .close()
        .map_err(|e| PyValueError::new_err(format!("fermeture Parquet échouée: {e}")))?;

    Ok(ParquetExport {
        out_path: out_path.to_string(),
        rows: n,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Nombre de coeurs logiques détectés sur la machine (pour vérifier que
/// l'adaptation matérielle fonctionne bien, sans rien coder en dur).
#[pyfunction]
fn detected_cpus() -> usize {
    num_cpus::get()
}

/// Module Python exposé.
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
