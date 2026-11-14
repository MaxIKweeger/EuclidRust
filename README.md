# euclid_fast

High-performance FITS I/O for Python, written in Rust and exposed via
[PyO3](https://pyo3.rs/) as a normal, `pip install`-able Python extension
module. Built as a testbed for data-processing tools for the ESA **Euclid**
mission, but useful for any FITS-based workflow.

```python
import euclid_fast

stats = euclid_fast.fits_stats("tile.fits")
cutout = euclid_fast.extract_cutout("tile.fits", ra_deg=83.82, dec_deg=-5.39, width=50, height=50)
euclid_fast.export_cutout_parquet("tile.fits", 83.82, -5.39, 50, 50, "cutout.parquet")
```

> **About this project.** The code, benchmarks and documentation here were
> written by [Claude Code](https://claude.com/claude-code) (Anthropic's coding
> agent). The author decided what to build, in what order, how to measure it
> and when to stop, and steered the agent throughout. Treat the project as
> experimental: read the code, run the benchmark yourself, and judge the
> numbers rather than the author.

**Contents:**
[What files it works with](#what-kind-of-files-does-it-work-with) ·
[How Python and Rust fit together](#how-python-and-rust-fit-together-pyo3) ·
[Install](#install) ·
[Use it](#use-it) ·
[Many images at once](#many-images-at-once) ·
[Tests](#run-the-tests-optional) ·
[Run the benchmark](#run-the-benchmark-on-your-own-fits-file) ·
[Troubleshooting](#troubleshooting) ·
[Motivation](#motivation) ·
[Benchmark results](#benchmark-results) ·
[Limitations](#known-limitations-read-before-relying-on-this-for-science) ·
[Roadmap](#roadmap)

## What kind of files does it work with?

**Short answer: 2-D FITS images, raw or tile-compressed (Rice / GZIP), whether the
image sits in the primary HDU or in an extension.**

FITS (Flexible Image Transport System) is the standard file format of
astronomy. A FITS file is a sequence of blocks called HDUs ("header/data
units"): each has a text header and a payload, which can be an image or a
table. `euclid_fast` reads **images**: a rectangular grid of pixels (a sky
mosaic, a detector frame...), plus the header keywords that say how pixels
map to the sky (the WCS).

Where the image sits depends on the product. A Euclid **MER mosaic** keeps
its image in the first HDU (the "primary HDU"), which is what `euclid_fast`
reads by default. **Calibrated frames** (VIS, NISP) and **raw frames** have
an *empty* primary HDU and one image extension per detector, with names such
as `DET11.SCI` or `1-1.E`. For those you pick the HDU with the `hdu=`
argument (an index or a name), and `list_hdus(path)` shows what a file
contains.

**Supported:**

| Requirement | Details |
|---|---|
| An image HDU | 2-D (`NAXIS = 2`). The primary HDU by default, or any image extension through `hdu=` (0-based index or `EXTNAME`) |
| Compression | None, or FITS tile compression with `RICE_1`, `GZIP_1` or `GZIP_2` (the `.fits.fz` format written by `fpack` and by astropy's `CompImageHDU`), including quantized floating-point images (`NO_DITHER`, `SUBTRACTIVE_DITHER_1`, `SUBTRACTIVE_DITHER_2`). A compressed image always lives in an extension, so pass `hdu=1` (or its name) |
| Pixel type (`BITPIX`) | `8`, `16`, `32`, `64` (integers), `-32`, `-64` (floating point). `BZERO` / `BSCALE` are applied, so unsigned 16-bit raw data (`BZERO = 32768`) works. 64-bit integers, signed or unsigned (FITS stores unsigned 64-bit as signed with `BZERO = 2^63`), are handled **exactly**, in raw and in gzip-compressed images (cfitsio compresses 64-bit integers with gzip only), including a whole-number `BSCALE` / `BZERO`: the exact values are in `min_int` / `max_int` (`fits_stats`) and `data_int` (`extract_cutout`), as Python ints, and the Parquet `value` column is `Int64` / `UInt64`; the ordinary float fields hold the nearest `float64` |
| WCS, only needed for cutouts | `CTYPE1/2` = `RA---xxx` / `DEC--xxx` with `xxx` one of `TAN`, `SIN`, `ARC`, `STG`, `ZEA`, `ZPN` (the zenithal projections), `TAN-SIP` (tangent plus the SIP polynomial distortion) or `TPV` (tangent plus the `PV1_m` / `PV2_m` polynomials used by the Euclid VIS pipeline), together with `CRVAL1/2`, `CRPIX1/2` and a linear matrix written as `CD` (`CD1_1`, `CD1_2`, `CD2_1`, `CD2_2`), as `CDELT1/2` with an optional `PC` matrix, or as `CDELT1/2` with the legacy `CROTA2`. `LONPOLE` is honoured. `fits_stats` needs no WCS |

**Not supported.** Anything else is rejected with a `ValueError` that says
why, rather than being read wrongly (these cases are covered by
`tests/test_input_validation.py`):

| Kind of file | Error you get |
|---|---|
| Tile compression with another algorithm (`HCOMPRESS_1`, `PLIO_1`) | `HDU 1 (SCI): tile-compressed image uses ZCMPTYPE = HCOMPRESS_1 which is not supported yet (supported: RICE_1, GZIP_1, GZIP_2)` |
| An extension that is a table (catalogue, spectrum...) | `HDU 1 is a BINTABLE table, not an image` |
| 3-D cubes (`NAXIS = 3`) | `NAXIS = 3 is not supported ...: only 2-D images are handled` |
| Not a FITS file | `truncated file: no END card found` |
| A projection that is not implemented (cylindrical or all-sky ones such as `CAR`, `AIT`, `MOL`, `HPX`...), a mismatched `CTYPE` pair, or a missing `CTYPE` | Cutouts raise `unsupported WCS projection: found CTYPE1/CTYPE2 = ...; supported are RA---TAN/DEC--TAN (also with -SIP distortion), RA---TPV/DEC--TPV and the zenithal projections SIN, ARC, STG, ZEA and ZPN ...` (`fits_stats` still works) |
| Distortion stored in lookup tables (HST `CPDIS` / `D2IMDIS` keywords) | Cutouts raise `this header declares distortion corrections stored in lookup tables ... which euclid_fast does not apply` (`fits_stats` still works) |
| No WCS at all (no `CRVAL` / `CRPIX` / matrix keywords) | Cutouts raise `this file has no WCS header (CRVAL, CRPIX and either a CD matrix or CDELT/PC)` (`fits_stats` still works) |

**Null pixels.** NaN pixels of floating-point images are left out of `min`, `max` and `mean` and counted in `n_nan` (infinities are values and stay in). The `BLANK` null value of integer images is **not** honoured: those pixels are counted as ordinary values.

If you forget `hdu=` on a file whose primary HDU is empty, the error lists
the image HDUs it found and tells you what to pass, for example
`[1] DET11.SCI (2040x2040, BITPIX -32) ... Pass hdu=<index or EXTNAME>`.

**What this means for Euclid Q1 data.** I read the headers of a handful of
files in each product family of the public Q1 release (this is a sample, not
a full survey of the archive) and ran the code on real files from four of
them, comparing with astropy:

| Product | Where the image is | Status |
|---|---|---|
| MER mosaics (`EUC_MER_BGSUB-MOSAIC-...`, uncompressed) | primary HDU | **Works** (stats and cutouts). All benchmark numbers come from the NISP Y-band tile. The other bands and the `RMS` / `FLAG` mosaics have the same layout but were not run |
| NISP calibrated frames (`EUC_NIR_W-CAL-IMAGE-BKG...`) | extensions `DETnn.SCI`, `.RMS`, `.DQ` (`TAN` WCS) | **Works** with `hdu="DET11.SCI"` etc. Checked on one detector: stats and a cutout identical to astropy |
| VIS calibrated frames (`EUC_VIS_SWL-BKG...`) | extensions `1-1.E`... (`TPV` WCS) | **Works** with `hdu="1-1.E"` etc. Checked on one detector: stats identical to astropy; over 300 random sky positions the cutout center agrees with astropy/wcslib to 3 × 10^-8 pixel, and all cutout pixels are identical |
| RAW frames, NISP (`EUC_LE1_NISP...`) | extensions `DETnn.SCI` (unsigned 16-bit via `BZERO`) | **Stats work** (checked on one detector). Cutouts are refused: the WCS is in detector-mosaic coordinates (`Y_MOSA`), not sky coordinates |
| RAW frames, VIS (`EUC_LE1_VIS...`) | extensions `1-1.E`... | Same layout as the other frames, **not run** |
| MER segmentation maps (`EUC_MER_FINAL-SEGMAP...`) | primary HDU, `BITPIX = 64` | **Stats work** (checked on the real 2.9 GB file: min and max identical to astropy, exact in `min_int` / `max_int`, mean agreeing to ~13 digits). No `CD` matrix in the header, so no cutouts |
| Catalogues, PSF files, other tables | table extensions | Not images, out of scope |

None of the Q1 image files I looked at were tile-compressed, so there was no compressed Euclid file to test on. Compressed-image support was validated on copies of real Euclid data that I compressed with astropy/cfitsio (see [Compressed images](#compressed-images)).

To check a file yourself before running anything on it, list its HDUs:

```python
import euclid_fast
for hdu in euclid_fast.list_hdus("file.fits"):
    print(hdu)
# <HduInfo index=0 kind=PRIMARY shape=- bitpix=8 readable=no (no image data)>
# <HduInfo index=1 name="DET11.SCI" kind=IMAGE shape=2040x2040 bitpix=-32 readable=yes>
```

or look at the header keywords with astropy:

```python
from astropy.io import fits

h = fits.getheader("file.fits", 1)          # 0 = primary HDU, 1 = first extension...
print("NAXIS", h["NAXIS"], "| BITPIX", h["BITPIX"])
print("CTYPE", h.get("CTYPE1"), h.get("CTYPE2"))
print("linear matrix keywords:", [k for k in ("CD1_1", "CDELT1", "PC1_1", "CROTA2") if k in h])
```

You want `NAXIS 2`, a `BITPIX` from the table above and, for cutouts, a `CTYPE`
pair from the projection list above and some linear-matrix keywords (`CD1_1`,
or `CDELT1`).

## How Python and Rust fit together (PyO3)

If you are an astronomer who knows Python but has never touched Rust, this
is the one thing to understand: **you never write or run Rust yourself.**
You install the package and `import` it like `numpy` or `astropy`.

Under the hood there are three pieces:

| Piece | Where | What it does |
|---|---|---|
| The Rust code | `euclid_fast/src/lib.rs` | Does the actual work: memory-maps the FITS file, runs the parallel computation, converts sky coordinates to pixels |
| **PyO3** | a dependency in `euclid_fast/Cargo.toml` | A Rust library that turns Rust functions and structs into Python functions and objects, and converts arguments between the two (for example, `hdu=` accepts a Python `int` or `str`, which PyO3 maps to a Rust `enum`) |
| **maturin** | `build-backend` in `euclid_fast/pyproject.toml` | Compiles the Rust into a shared library and installs it in your Python environment (`pip` calls it for you) |

PyO3 shows up in `lib.rs` as a handful of annotations. Here is the whole
mapping, taken from the real code:

```rust
#[derive(FromPyObject)]                         // -> accepts a Python int OR str
enum HduSelector { Index(usize), Name(String) }

#[pyfunction]                                   // -> a Python function
fn fits_stats(path: &str, hdu: Option<HduSelector>) -> PyResult<FitsStats> { /* ... */ }

#[pyclass]                                      // -> a Python class
struct FitsStats {
    #[pyo3(get)] min: f64,                      // -> readable attribute: stats.min
    #[pyo3(get)] max: f64,
    #[pyo3(get)] mean: f64,
    /* ... */
}

#[pymodule]                                     // -> `import euclid_fast`
fn euclid_fast(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fits_stats, m)?)?;
    /* ... */
    Ok(())
}
```

and on the Python side it is just:

```python
import euclid_fast                       # loads the compiled Rust library
s = euclid_fast.fits_stats("tile.fits")  # calls the Rust function
print(s.min, s.max, s.mean)              # reads the fields of the Rust struct
```

So the build chain is: **Rust source -> (maturin, run by pip) -> compiled
library such as `euclid_fast.cpython-313-x86_64-linux-gnu.so` (`.pyd` on
Windows) -> installed into your virtual environment -> `import euclid_fast`.**
The scripts in `scripts/` contain no Rust and no PyO3 code; they only
`import euclid_fast`. Errors raised in Rust arrive in Python as ordinary
`ValueError` exceptions.

## Install

### 1. Prerequisites

You need three things on your machine:

- **Python 3.8 or newer** (developed and tested with 3.13).
- **A Rust toolchain, version 1.88 or newer** (the Arrow / Parquet dependencies require it; developed with 1.100). Install it with [rustup](https://rustup.rs/):
  - Linux / macOS: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  - Windows: download `rustup-init.exe` from <https://rustup.rs/>. It will
    ask you to install the Visual Studio C++ Build Tools; accept.
  - Then **open a new terminal** and check with `rustc --version`.
- **A C linker**, which is normally already there (`build-essential` on
  Debian/Ubuntu, Xcode command-line tools on macOS, the Build Tools on
  Windows).

Everything else (maturin, PyO3 and the other Rust libraries) is downloaded
automatically during the build, so the first install needs internet access.

> Tested on Debian 13 (x86_64) with Python 3.13 and Rust 1.100, on an Intel Core
> i7-8559U (4 cores / 8 threads) with 31 GiB of RAM. macOS and
> Windows have not been tested; if you try them, please open an issue with
> what happened.

### 2. Get the code and install

```bash
git clone https://github.com/MaxIKweeger/Euclid.git
cd Euclid

python3 -m venv .venv                 # a private Python environment
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install ./euclid_fast             # compiles the Rust code (a few minutes)
```

The first compile takes a few minutes because the release profile enables
full link-time optimisation; later rebuilds are faster. When it finishes,
check that it worked:

```bash
python -c "import euclid_fast; print('OK, threads available:', euclid_fast.detected_cpus())"
```

If that prints `OK, threads available: <a number>`, you are done.

*Developing the Rust code?* Use `pip install maturin`, then run
`maturin develop --release` inside `euclid_fast/`. That installs the module
in editable mode and rebuilds it after you change `src/lib.rs`.

## Use it

You need a FITS file of the kind described in
[What kind of files does it work with?](#what-kind-of-files-does-it-work-with).
Euclid MER mosaics are of this kind; see
[the benchmark section](#run-the-benchmark-on-your-own-fits-file) for where
to get one.

```python
import numpy as np
import euclid_fast

path = "tile.fits"

# 0. What is in the file? (reads the headers only, instant)
for hdu in euclid_fast.list_hdus(path):
    print(hdu)

# 1. Global statistics over the whole image
s = euclid_fast.fits_stats(path)
print(s.min, s.max, s.mean)          # physical values (BZERO/BSCALE applied)
print(s.naxis1, s.naxis2, s.bitpix)  # image shape and data type
print(s.threads_used, s.elapsed_ms)  # how it ran

# 2. A cutout around a sky position (RA/Dec in degrees)
c = euclid_fast.extract_cutout(path, ra_deg=273.0716597, dec_deg=68.0,
                               width=50, height=50)
image = np.array(c.data).reshape(c.height, c.width)   # row-major -> 2-D array
print(c.pixel_x_center, c.pixel_y_center)             # FITS 1-indexed pixel position

# 3. The same cutout written to a Parquet file
r = euclid_fast.export_cutout_parquet(path, 273.0716597, 68.0, 50, 50, "cutout.parquet")
print(r.rows, r.out_path)

import pandas as pd                  # pip install pandas pyarrow
print(pd.read_parquet("cutout.parquet").head())       # columns: x, y, value
```

**Files whose images are in extensions** (Euclid calibrated and raw frames)
need the `hdu` argument, a name or a 0-based index, on all three functions:

```python
frame = "EUC_NIR_W-CAL-IMAGE-BKG_H-2681-0_20240930T175032.222263Z.fits"

s = euclid_fast.fits_stats(frame, hdu="DET11.SCI")              # by EXTNAME
s = euclid_fast.fits_stats(frame, hdu=1)                        # by index (same HDU)
c = euclid_fast.extract_cutout(frame, 266.09, 67.32, 50, 50, hdu="DET11.SCI")
```

(names are matched case-insensitively; if several HDUs share a name, the
first one is used).

**Compressed files** (`.fits.fz`, as written by `fpack`) work the same way: the image is in extension 1, so pass `hdu=1`.

```python
stats = euclid_fast.fits_stats("image.fits.fz", hdu=1)
cutout = euclid_fast.extract_cutout("image.fits.fz", 150.0, 2.2, 50, 50, hdu=1)   # decodes only the tiles it needs
```

Lossy (quantized) floating-point images decode to exactly the values astropy/cfitsio would give, which are not the original pre-compression values; that loss belongs to the format, not to `euclid_fast`.

Reference:

| Function | Returns | Fields |
|---|---|---|
| `fits_stats(path, hdu=None)` | `FitsStats` | `min`, `max`, `mean`, `naxis1`, `naxis2`, `bitpix`, `threads_used`, `elapsed_ms`, and `min_int`, `max_int` (exact integers for 64-bit integer images, otherwise `None`), `n_nan` (number of NaN pixels, which `min`, `max` and `mean` leave out) |
| `extract_cutout(path, ra_deg, dec_deg, width, height, hdu=None)` | `Cutout` | `data` (flat list of floats, row-major), `data_int` (the same pixels as exact Python ints for 64-bit integer images, otherwise `None`), `width`, `height`, `pixel_x_center`, `pixel_y_center`, `elapsed_ms` |
| `export_cutout_parquet(path, ra_deg, dec_deg, width, height, out_path, hdu=None)` | `ParquetExport` | `out_path`, `rows`, `elapsed_ms` |
| `fits_stats_many(paths, hdu=None, skip_errors=False)` | list of `FitsStats` (or `None`) | one per file, in the same order; see [Many images at once](#many-images-at-once) |
| `fits_stats_all_hdus(path, name_contains=None)` | list of `HduStats` | per image HDU: `index`, `name`, `naxis1`, `naxis2`, `bitpix`, `min`, `max`, `mean`, `min_int`, `max_int`, `n_nan`, `elapsed_ms` |
| `extract_cutouts(path, ra_deg, dec_deg, width, height, hdu=None, skip_errors=False)` | list of `Cutout` (or `None`) | one per position, same fields as `extract_cutout` |
| `list_hdus(path)` | list of `HduInfo` | per HDU: `index`, `kind` (`PRIMARY`, `IMAGE`, `BINTABLE`...), `name`, `bitpix`, `shape` (NAXIS1, NAXIS2...), `readable`, `note` (why it is not readable) |
| `detected_cpus()` | `int` | number of logical cores found on this machine |

Notes: `hdu=None` (the default) reads the primary HDU, exactly like before.
`Cutout.data` is a plain Python list, so convert it with `np.array(...)` as
above (fine for cutouts; it is not meant for multi-megapixel regions). The
Parquet file has one row per pixel with columns `x`, `y` (absolute pixel
coordinates in the source image) and `value` (an exact `Int64` / `UInt64` column for 64-bit integer images, `Float64` otherwise). All errors (file not found,
wrong kind of HDU, cutout outside the image, unsupported WCS...) are raised
as `ValueError`.

## Many images at once

When you have many images, three functions do the looping in Rust, in
parallel on all the cores, with the GIL released:

```python
# 1. The same statistics over many files (one result per file, same order)
results = euclid_fast.fits_stats_many(paths)
results = euclid_fast.fits_stats_many(paths, hdu=1)               # hdu applies to every file
results = euclid_fast.fits_stats_many(paths, skip_errors=True)    # None for a file that cannot be read

# 2. Every image HDU of one frame (all the detectors of a Euclid frame)
for r in euclid_fast.fits_stats_all_hdus("frame.fits", name_contains="SCI"):
    print(r.index, r.name, r.min, r.max, r.mean)

# 3. Many cutouts from one file, e.g. a catalogue of objects
#    (ras, decs: two equal-length lists or numpy arrays, in degrees)
cutouts = euclid_fast.extract_cutouts("tile.fits", ras, decs, 50, 50)
images = [np.array(c.data).reshape(50, 50) for c in cutouts]
```

Details:

- Results come back in the order of the inputs. `fits_stats_all_hdus`
  returns the HDUs in file order, skipping the ones that are not readable
  images (the empty primary HDU, tables...); `name_contains` keeps only the
  HDUs whose `EXTNAME` contains that text, case-insensitively (for example
  `"SCI"` to leave out the `RMS` and `DQ` images).
- By default the first problem raises a `ValueError` that names the file (or
  the position, for cutouts). With `skip_errors=True`, the failed items are
  `None` instead, so you can process a whole list and look at the failures
  afterwards.
- `extract_cutouts` opens the file and reads its header once, then
  extracts the positions in parallel.
- **The GIL is released** during all the calls (single or batch), so other
  Python threads keep running while Rust works. For example, a counter in the
  main thread reached 731,463 while one 155 ms `fits_stats` call on the 1.47
  GB mosaic ran in another thread.

What this buys you, measured on real Euclid data (median of 3 runs, warm
cache; the 40-file case uses 40 copies of one detector, so everything is in
the OS cache):

| Task | `astropy` | `euclid_fast`: a loop of single calls | `euclid_fast`: one batch call |
|---|---|---|---|
| Stats of the 48 image HDUs of a NISP frame (799 MB) | 484 ms | 102 ms | 72 ms |
| Stats of the 144 image HDUs of a VIS frame (2.4 GB) | 1309 ms | 693 ms | 299 ms |
| Stats of 40 files (17 MB each, uncompressed) | 274 ms | 68 ms | 55 ms |
| Stats of 40 Rice-compressed files | 3317 ms | 583 ms | 572 ms |
| 1000 cutouts 50×50 from the 1.47 GB mosaic | 108 ms (WCS built once) / 2835 ms (WCS and file reopened each time) | 216 ms | 39 ms |

Every result is identical to astropy's and to the single calls'. How to read
the table: the batch call helps most where each item is small (many cutouts,
frames with many small HDUs). On big or compressed files a single call
already uses all the cores, so a batch call is barely faster than a loop;
the large gain over astropy there comes from reading and decoding in
parallel, not from batching.

## Run the tests (optional)

`tests/test_input_validation.py` builds small FITS files with astropy and
checks that supported files work (primary HDU, extensions, `BITPIX` 16 and
64 including exact unsigned 64-bit values and gzip-compressed 64-bit images, `BZERO`/`BSCALE`), that unsupported ones are rejected with a clear error,
that every supported projection, matrix form, the TPV distortion and the SIP
distortion (including real SIP headers from astropy's own test data) agree with
astropy/wcslib, and that tile-compressed images (`RICE_1`, `GZIP_1`,
`GZIP_2`, quantization and dithering) decode exactly like astropy/cfitsio, and
that the batch functions return what the single calls return. It needs no download. With `euclid_fast`
installed and `pip install -r scripts/requirements.txt` done:

```bash
python tests/test_input_validation.py     # prints "all checks passed"
```

(`pytest tests/` works too.)

## Run the benchmark on your own FITS file

The script `scripts/benchmark_vs_astropy_real.py` times `euclid_fast`
against `astropy` on a file of your choice, and checks that both give the
same answer.

**1. Get a FITS file.** Use any file that matches the
[requirements above](#what-kind-of-files-does-it-work-with). To reproduce the
numbers in this README, use the Euclid Q1 tile they were measured on, from
the public IRSA mirror (anonymous S3 bucket `nasa-irsa-euclid-q1`, no account
needed, 1.47 GB):

<https://nasa-irsa-euclid-q1.s3.amazonaws.com/q1/MER/102160339/NISP/EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE102160339-3A9BC7_20241024T231706.521789Z_00.00.fits>

Download it with your browser, `curl` or `wget`, and put it anywhere. To try
a **calibrated frame** instead (images in extensions), these are public too:

- NISP, 799 MB (`TAN` WCS): <https://nasa-irsa-euclid-q1.s3.amazonaws.com/q1/NIR/2681/EUC_NIR_W-CAL-IMAGE-BKG_H-2681-0_20240930T175032.222263Z.fits>
- VIS, 2.4 GB (`TPV` WCS): <https://nasa-irsa-euclid-q1.s3.amazonaws.com/q1/VIS/2681/EUC_VIS_SWL-BKG-002681-00-1-0000000__20241017T042839.727792Z.fits>

**2. Install the benchmark's two dependencies** (inside the same virtual
environment you installed `euclid_fast` in):

```bash
pip install -r scripts/requirements.txt     # numpy + astropy
```

**3. Run it**, giving the path to the file:

```bash
python scripts/benchmark_vs_astropy_real.py /path/to/tile.fits
python scripts/benchmark_vs_astropy_real.py /path/to/frame.fits --hdu DET11.SCI   # image in an extension
```

Optional flags: `--hdu` (the HDU holding the image, an `EXTNAME` or a 0-based
index; default: the primary HDU), `--ra` / `--dec` (cutout center in degrees;
default: the sky position of the image center, taken from the file's own
WCS), `--size` (cutout width and height in pixels, default 50) and `--runs`
(repetitions per measurement, default 5). If the HDU is tile-compressed, astropy has to decompress the whole image for the global stats and reads the cutout through `.section`; the script handles both. If the file has no sky WCS (raw
detector frames, for example), the cutout comparison is skipped with an
explanation and the global-stats comparison still runs. If the file uses
`BZERO`/`BSCALE`, astropy cannot memory-map it and reads it into RAM; the
script says so, and the comparison is then less favourable to astropy than
usual for reasons unrelated to `euclid_fast`.

**4. Read the output.** It prints, for each of the two operations, the median
time of both tools, the speedup, and a correctness check:

```
--- 1. Global stats (min/max/mean over the whole file) ---
astropy (memmap, numpy)      :   476.49 ms
euclid_fast (Rust/rayon)     :   130.05 ms
Speedup                      : ×3.66  (72.7% less time)
Result check: astropy min/max/mean = (-111.43283081054688, 15220.939453125, 1.0181564092636108)
              rust    min/max/mean = (-111.43283081054688, 15220.939453125, 1.0181568676826722)

--- 2. 50x50 cutout extraction by RA/Dec (273.071586, 68.000028) ---
astropy (WCS + memmap)       :    2.670 ms
euclid_fast (Rust/mmap)      :    0.224 ms
Speedup                      : ×11.91  (91.6% less time)
...
              all 2500 pixels identical: True
```

(That is a real run on the development machine; your times will differ with
your hardware, and a re-run on the same machine moves by a few percent.)
`min`/`max` and every cutout pixel should match exactly; `mean` differs in
the 7th significant figure because the two tools sum floats in a different
order. Run it twice: the first run may be slower while the operating system
loads the file into its cache.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `rustc: command not found` or pip fails with "Cargo, the Rust package manager, is not installed" | Rust is not installed, or the terminal was opened before installing it. Install via [rustup](https://rustup.rs/) and open a **new** terminal. |
| Build error mentioning `edition2024` or `rust-version` | Your Rust is older than 1.88. Run `rustup update`. |
| `linker 'cc' not found` (Linux) / `link.exe not found` (Windows) | Install a C toolchain: `sudo apt install build-essential`, or the Visual Studio C++ Build Tools. |
| `ModuleNotFoundError: No module named 'euclid_fast'` | You are in a different Python than the one you installed into. Activate the virtual environment (`source .venv/bin/activate`) and retry; check with `pip show euclid_fast`. |
| `ValueError: the primary HDU holds no image (NAXIS = 0); it contains: ...` | The image is in an extension (Euclid calibrated/raw frames). Pass `hdu=` with the name or index shown in the message. |
| `ValueError: no HDU named '...'` / `no HDU with index ...` | Wrong `hdu` value. The message lists the available HDUs; `euclid_fast.list_hdus(path)` shows them all. |
| `ValueError: ... tile-compressed image uses ZCMPTYPE = ...` or `... is a BINTABLE table, not an image` | That HDU uses a compression algorithm that is not implemented (only `RICE_1`, `GZIP_1`, `GZIP_2` are), or it is a table. |
| `ValueError: NAXIS = 3 is not supported` or `no END card found` | A 3-D cube, or a file that is not FITS. Only 2-D images are handled. |
| `ValueError: unsupported WCS projection ...` | The cutout functions implement the zenithal projections (`TAN`, `SIN`, `ARC`, `STG`, `ZEA`, `ZPN`), `TAN-SIP` and `TPV`. Cylindrical and all-sky projections, and WCS in other coordinate systems (galactic, ecliptic), are not supported yet. |
| `ValueError: this header declares distortion corrections stored in lookup tables ...` | The header has HST-style `CPDIS` / `D2IMDIS` keywords. Their correction tables are not applied, so cutouts are refused. `fits_stats` still works. |
| `ValueError: this file has no WCS header` | The cutout functions need `CRVAL1/2`, `CRPIX1/2` and a linear matrix (`CD`, or `CDELT` with an optional `PC`) in the chosen HDU. `fits_stats` still works without them. |
| `ValueError: cutout outside the image` | The RA/Dec you asked for falls outside the image. Check the coordinates, and for frames make sure `hdu=` is the detector you mean (each detector covers a different patch of sky). |
| The benchmark exits asking for `numpy`/`astropy` | Run `pip install -r scripts/requirements.txt` in the same environment. |

## Motivation

Euclid will deliver petabytes of high-resolution imaging and catalogue
data over its six-year survey. The standard astronomy stack for this kind
of workload — `astropy.io.fits` on top of the C library `cfitsio` — has
two structural limitations that become painful at this scale, especially
for independent researchers or small labs without access to the official
consortium's HPC clusters:

- **The GIL.** Even on a machine with 8 logical cores, a single Python
  thread holds the Global Interpreter Lock while `cfitsio` does its work,
  so "read 1000 FITS files in parallel" rarely scales the way you'd hope
  from thread count alone.
- **Unnecessary copies.** Reading a single cutout out of a multi-gigabyte
  tile often pulls far more data into memory than the cutout itself
  requires.

`euclid_fast` addresses both with a small Rust core: `mmap`-based
zero-copy access to uncompressed FITS data, and a `rayon` thread pool that
auto-sizes itself to the host's logical core count (the GIL is released while it works) — there is no hardcoded
thread count, so the same binary scales itself from a laptop to a
many-core workstation.

## What it does today

Three single-image operations (plus `list_hdus` to see what a file contains, and batch versions of them, see [Many images at once](#many-images-at-once)), each independently benchmarked against `astropy` on
identical input and compared for numerical correctness, not just speed:

| Function | What it computes | Notes |
|---|---|---|
| `fits_stats(path, hdu=None)` | Global `min` / `max` / `mean` over one image HDU (the primary one by default) | `mmap` + `rayon`, chunked reduction tuned for this workload |
| `extract_cutout(path, ra_deg, dec_deg, width, height, hdu=None)` | Pixel-space cutout around a sky position | Zenithal WCS (TAN, SIN, ARC, STG, ZEA, ZPN), with TPV or SIP distortion; FITS 1-indexed convention |
| `export_cutout_parquet(path, ra_deg, dec_deg, width, height, out_path, hdu=None)` | Same cutout, written as Parquet | One row per pixel: columns `x`, `y` (absolute pixel coords in the source image), `value` (physical value, `BZERO`/`BSCALE` applied) |

## Benchmark results

All numbers are measured on a **real Euclid Quick Data Release 1 (Q1)
product** from the public IRSA mirror (see
[Run the benchmark](#run-the-benchmark-on-your-own-fits-file) for the exact
file):

- **File**: `EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE102160339-...fits` — a NISP
  Y-band, background-subtracted MER mosaic tile near the Euclid Deep
  Field North. 19200×19200 px, `float32` (`BITPIX = -32`), 1.47 GB.
- **WCS**: standard `RA---TAN`/`DEC--TAN` with a `CD` matrix and no SIP
  terms.

Measured on the development machine this project was built on: an Intel
Core i7-8559U (4 cores / 8 threads, 2.7 GHz base, up to 4.5 GHz), 31 GiB of RAM, an NVMe SSD, no discrete GPU —
a laptop-class machine, not a cluster. This matters: the point of these
numbers is "what you get on hardware a working astronomer might actually
have on their desk," not a best-case HPC result. Each number is the
median of 5 runs after a warm-up pass, so the OS page cache is in a
comparable state for both tools.

| Operation | `astropy` | `euclid_fast` | Speedup | Time saved |
|---|---|---|---|---|
| Global stats (min/max/mean, full 1.47 GB) | 478.30 ms | 130.35 ms | ×3.67 | −72.7% |
| Cutout extraction (50×50 px, by RA/Dec) | 2.664 ms | 0.219 ms | ×12.16 | −91.8% |

Re-runs of the same script on the same machine land within a few percent
of these (for example ×3.65 and ×12.36).

`min` and `max` matched bit-for-bit between the two implementations.
`mean` matched to 7 significant figures (1.0181564... vs 1.0181568...) —
the residual difference is floating-point summation order (NumPy's
pairwise summation vs. this project's parallel tree reduction;
floating-point addition is not strictly associative), not a computational
error. The cutout pixels are identical.

Two things are worth noting about *why* the gap differs between the two
operations:

- The **cutout** benchmark shows the larger relative gain, because it is
  dominated by per-call Python/WCS/NumPy object overhead — exactly the
  overhead this project is designed to remove. The benchmark builds the
  `WCS` object inside every call, like a naive script would; an astropy
  user who builds it once and reuses it will see a smaller gap.
- The **global stats** benchmark is ultimately bounded by memory
  bandwidth, not CPU: scanning a full tile moves more than a gigabyte
  through RAM regardless of which language does the summing, so the
  ceiling on the achievable speedup is lower and both tools get closer to
  it.

### Compressed images

Tile-compressed images (`RICE_1`, `GZIP_1`, `GZIP_2`, with quantized or
lossless floats) are decoded tile by tile, with the tiles spread over the
threads; a cutout decodes only the tiles it touches. Results are checked
against astropy (cfitsio): min/max and every cutout pixel are identical, and
`mean` agrees to ~10^-9 relative, over 40 test configurations (algorithms,
pixel types, tile shapes, dither methods and seeds, NaN pixels, unsigned
16-bit) and on the real Euclid data below.

I had no Euclid-produced compressed file (none of the Q1 images I checked
were compressed), so the real-data runs use real Euclid images that I
compressed myself with astropy's `CompImageHDU` (cfitsio underneath), with
its defaults (quantization level 16, dithering) unless stated:

| Real Euclid data, compressed with astropy | Global stats: `astropy` -> `euclid_fast` | 50×50 cutout: `astropy` (`.section`) -> `euclid_fast` |
|---|---|---|
| NISP detector, 2040², `RICE_1` | 78.6 ms -> 14.5 ms (×5.4) | 6.96 ms -> 0.54 ms (×12.9) |
| NISP detector, `GZIP_2` | 219 ms -> 22.4 ms (×9.8) | 10.5 ms -> 0.80 ms (×13.2) |
| NISP detector, `GZIP_2` lossless | 176 ms -> 12.8 ms (×13.7) | 8.58 ms -> 0.48 ms (×17.8) |
| VIS detector, 2048×2066, `RICE_1` | 72.7 ms -> 11.2 ms (×6.5) | 7.37 ms -> 0.46 ms (×15.9) |
| MER mosaic, 19200², `RICE_1` (1.47 GB -> 308 MB) | 3719 ms -> 1216 ms (×3.06) | 14.96 ms -> 3.71 ms (×4.0) |

Median of 3 runs on the development machine, warm cache. For `astropy`, the
global stats include decompressing the whole image (single-threaded), which
is what an astropy user does; the gap is wider than for uncompressed files
because `euclid_fast` decompresses tiles on all the cores. On the big mosaic
the tiles are whole image rows (19200 pixels wide), so a cutout still has to
decode about fifty full rows, which is why the cutout gain is smaller there.
These are decoding speeds only: the files were produced by cfitsio, and
Euclid's own compressed products (if any) may use other settings.

### How the Rust side got there: an honest log of what worked

The speedups above are the result of iterating against measurements, not
a single clever trick:

1. **Naive version** — one `rayon` task per pixel via `into_par_iter()`.
   Correct, but the per-item scheduling overhead dominates.
2. **Chunked reduction** — split the buffer into 65,536-pixel blocks,
   each reduced by a tight sequential loop (which LLVM auto-vectorizes at
   `opt-level = 3` without any explicit SIMD intrinsics), with `rayon`
   only parallelizing across blocks. This cut the stats time by more than
   half (×2.24 on a ~1 GB file during development).
3. **mmap vs. eager read** — reading the whole file into a `Vec<u8>`
   first was measured to be **1.87× slower** than `mmap`, confirming that
   letting the OS page cache serve data on demand beats an eager bulk
   copy for this access pattern.
4. **Stopping point** — once (2) and (3) were in place, further
   micro-optimization (explicit SIMD intrinsics, finer chunk tuning)
   showed diminishing returns against the memory-bandwidth ceiling
   identified above. That's the deliberate stopping condition for this
   round of tuning, not an oversight.

Build settings: `cargo` release profile with `lto = "fat"`,
`codegen-units = 1`, `opt-level = 3`. Thread pool size is read at runtime
from the host (`rayon::current_num_threads()` / `num_cpus::get()`) —
never hardcoded, so the same wheel adapts to whatever machine runs it.

## Known limitations (read before relying on this for science)

### What it reads

- **2-D images only.** No tables or catalogues, no 3-D cubes, no random-groups
  files. Each call reads one HDU of one file (the batch functions loop over
  files, HDUs or positions for you).
- **Compression: `RICE_1`, `GZIP_1` and `GZIP_2` only.** `HCOMPRESS_1` and
  `PLIO_1` are not implemented and are refused with a clear error. The
  compressed files tested were produced by astropy/cfitsio, not by Euclid's
  pipeline.
- **WCS coverage.** Zenithal projections (`TAN`, `SIN`, `ARC`, `STG`, `ZEA`,
  `ZPN`), with TPV or SIP distortion. Not supported: cylindrical and all-sky
  projections (`CAR`, `AIT`, `MOL`, `HPX`...), coordinate systems other than
  RA/Dec, alternate WCS (`A`-`Z`: only the primary one is read), and
  distortion given by lookup tables (HST `CPDIS` / `D2IMDIS`, which makes
  cutouts fail with an error). The sky frame is not read: `RA` and `Dec` are
  taken in the file's own frame, and `RADESYS` / `EQUINOX` are neither read
  nor converted.
- **Null pixels.** NaN pixels are left out of the statistics; the integer
  `BLANK` null value is ignored (see above).

### What a cutout is

- **A cutout is a crop, not a resampling.** It is a rectangle of whole
  pixels whose corner is `round(centre - size / 2)` (halves round away from
  zero, whereas Python's `round` rounds halves to even, so a hand-made
  comparison can differ by one pixel on an exact tie). The requested position
  can therefore be up to half a pixel from the middle of the crop; the exact
  fractional position is returned in `pixel_x_center` / `pixel_y_center`.
  There is no interpolation, rotation or sub-pixel shift, and sizes are in
  pixels, not in angle.
- **Cutouts must lie fully inside the image.** One that touches the edge or
  extends beyond it is refused (`cutout outside the image`); there is no
  padding and no partial cutout.
- **`Cutout.data` is a plain Python list of floats** (and `data_int` a list
  of ints): fine for thumbnails, not meant for multi-megapixel regions.

### Numbers

- **Statistics.** `mean` is the mean of the valid (non-NaN) pixels. It
  agrees with numpy to about 7 significant figures on float32 images, not to
  the last digit, because the sums are made in a different order. Images with
  NaN pixels take about three times longer than clean ones, because a second,
  NaN-aware pass is needed (still faster than numpy's `nanmin` + `nanmax` +
  `nanmean`; measured: 70 ms against 164 ms on 256 MB with 7% NaN).
- **64-bit integers are exact only in the integer fields.** For 64-bit
  integer images (signed, or unsigned via `BZERO = 2^63`; raw or
  gzip-compressed) the exact values are `min_int` / `max_int` and
  `data_int`, and the Parquet `value` column. The float fields (`min`, `max`,
  `mean`, `data`) are `float64`, exact only up to 2^53 (about 9 × 10^15),
  which is inherent to the type. The exact path needs whole-number `BSCALE`
  (|BSCALE| ≤ 2^20) and `BZERO` (|BZERO| ≤ 2^64); with a fractional `BSCALE`
  or `BZERO` the physical values are not integers anyway, so everything is
  `float64`.
- **The sky-to-pixel inverse of TPV and SIP is iterative.** It was checked
  against wcslib (through astropy) on over 120 synthetic headers and on real
  ones: the real Euclid VIS frame (TPV, 3 × 10^-8 pixel) and the real SIP
  headers shipped with astropy's tests, including HST/ACS (4 × 10^-9 pixel
  at worst). It can pick the wrong root when a SIP distortion is so strong
  (hundreds of pixels) that the inverse is not unique, which does not happen
  with real instruments.

### How far it has been checked

- **Real-data validation is small.** The benchmark numbers above come from
  a single Q1 NISP mosaic tile out of the ~35 TB Q1 release. Correctness
  (not speed) was also checked against astropy on one NISP frame, one VIS
  frame, one raw NISP detector and one segmentation map. That is enough to
  show the approach works on real archival products, not a representative
  sample across instruments, bands and tiles.
- **Tested on one machine only.** An Intel NUC running Debian 13 (Linux
  6.12, x86_64): Intel Core i7-8559U (4 cores / 8 threads, 2.7 GHz, up to
  4.5 GHz), 31 GiB of RAM, an NVMe SSD, Python 3.13.5 and Rust 1.100. macOS,
  Windows and ARM have not been tried, nor other Python versions (the
  package declares 3.8 or newer, which is what PyO3 supports, but only 3.13
  was run), nor machines with a very different core count or much less
  memory (the images are memory-mapped, so a laptop with 8 GB of RAM should
  cope with the 1.47 GB tile, but I have not measured it).
- **No continuous integration, no pre-built packages.** The tests are run by
  hand. The workflow file in `euclid_fast/.github/` is not at the root of the
  repository, so GitHub never runs it. The package is not on PyPI: you build
  it yourself, which needs a Rust toolchain. The interface is version 0.1.0
  and may change.

### Practical points

- **The files are memory-mapped and must not change while they are read.**
  If another process truncates or overwrites a file during a call, the
  operating system can kill the Python process (this is how `mmap` works; I
  have not tried to provoke it). Network or FUSE file systems are untested.
- **All the cores are used by default.** The number of threads follows the
  machine; to limit it (a shared node, for example) set the environment
  variable `RAYON_NUM_THREADS` before starting Python, e.g.
  `RAYON_NUM_THREADS=4`. The GIL is released during the calls.
- **Memory with compressed images.** Each worker thread decodes whole tiles
  into `float64` (8 bytes per pixel), so an image compressed as a single huge
  tile needs about eight times its pixel count in bytes. There is no tile
  cache: many cutouts falling in the same tile decode it again each time.
- **Batch results are held in memory.** `fits_stats_many` and
  `extract_cutouts` return everything as Python lists at the end: there is
  no streaming and no progress callback, so a huge list of cutouts uses
  memory proportional to its size. `hdu=` is the same for every file of a
  `fits_stats_many` call.

## Roadmap

This is the first of four planned components for Euclid-scale data
processing, chosen first specifically because it needed no GPU/cluster
and is the foundation the others build on:

1. ✅ **FITS streaming / cutout / Parquet export** (this package). Next steps for this component, if someone needs them: the remaining compression algorithms (`HCOMPRESS_1`, `PLIO_1`) and the cylindrical / all-sky projections.
2. Catalogue cross-matching at the hundred-million-to-billion-object
   scale (HEALPix / spherical KD-tree indexing, memory-mapped for
   datasets larger than available RAM).
3. Morphology / weak-lensing profile fitting (PyO3-bound, parallel) —
   feasible at prototype scale on commodity hardware, not yet at
   billion-galaxy scale without a cluster.
4. Interactive WASM/WebGPU visualization of large mosaics and catalogues.

## License / status

MIT licensed (see `LICENSE`) — free to use, copy, modify, merge, publish,
distribute and sublicense, with no restriction beyond keeping the
copyright notice. Experimental, actively developed.