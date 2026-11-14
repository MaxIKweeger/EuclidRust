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
[Tests](#run-the-tests-optional) ·
[Run the benchmark](#run-the-benchmark-on-your-own-fits-file) ·
[Troubleshooting](#troubleshooting) ·
[Motivation](#motivation) ·
[Benchmark results](#benchmark-results) ·
[Limitations](#known-limitations-read-before-relying-on-this-for-science) ·
[Roadmap](#roadmap)

## What kind of files does it work with?

**Short answer: one specific kind of FITS file, a plain 2-D image.**

FITS (Flexible Image Transport System) is the standard file format of
astronomy. A FITS file is a sequence of blocks called HDUs ("header/data
units"): each has a text header and a payload, which can be an image or a
table. `euclid_fast` reads **images**: a rectangular grid of pixels (a
sky mosaic, for example) stored in the **first HDU** of the file (the
"primary HDU"), plus the header keywords that say how pixels map to the sky
(the WCS). The reference example is a Euclid **MER mosaic tile**
(`EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE...fits`, a 19200 × 19200 float32 image),
which is exactly what the benchmarks use.

**Supported:**

| Requirement | Details |
|---|---|
| Uncompressed image in the primary HDU | A 2-D image (`NAXIS = 2`), with the pixels in the first HDU of the file |
| Pixel type (`BITPIX`) | `8`, `16`, `32` (integers), `-32`, `-64` (floating point). `BZERO` / `BSCALE` are applied |
| WCS, only needed for cutouts | `CTYPE1/2 = RA---TAN / DEC--TAN`, with `CRVAL1/2`, `CRPIX1/2` and the `CD` matrix (`CD1_1`, `CD2_2`, and `CD1_2`, `CD2_1` if rotated). `fits_stats` needs no WCS |

**Not supported.** Anything else is rejected with a `ValueError` that says
why, rather than being read wrongly (these cases are covered by
`tests/test_input_validation.py`):

| Kind of file | Error you get |
|---|---|
| Compressed images (Rice / GZIP tiles, `.fits.fz`), or an image stored in an extension rather than the primary HDU (multi-extension FITS), including tables and catalogues | `the primary HDU holds no image (NAXIS = 0): the image is probably stored in an extension or is tile-compressed ...` |
| 3-D cubes (`NAXIS = 3`) | `NAXIS = 3 is not supported: only 2-D images are handled` |
| `BITPIX = 64` | `BITPIX 64 is not supported yet` |
| Not a FITS file | `truncated file: no END card found` |
| A projection other than TAN (`SIN`, `ARC`...), TAN with SIP distortion (`CTYPE` ending in `-SIP`), or a missing `CTYPE` | Cutouts raise `unsupported WCS projection: expected CTYPE1/CTYPE2 = RA---TAN/DEC--TAN, found ...` (`fits_stats` still works) |
| WCS written with `CDELT` (+`PC`) instead of a `CD` matrix, or no WCS at all | Cutouts raise `this file has no WCS header` (`fits_stats` still works) |

To check a file before running anything on it, look at its header with
astropy:

```python
from astropy.io import fits

h = fits.getheader("file.fits", 0)                  # header of the primary HDU
print("NAXIS", h["NAXIS"], "| BITPIX", h["BITPIX"])
print("CTYPE", h.get("CTYPE1"), h.get("CTYPE2"))
print("CD matrix present:", "CD1_1" in h and "CD2_2" in h)
```

You want `NAXIS 2`, a `BITPIX` from the table above, `CTYPE RA---TAN
DEC--TAN` and `CD matrix present: True`. If `NAXIS` is `0`, your image is in
an extension or compressed, which is not supported yet.

On Euclid data, the files that fit are the uncompressed **MER mosaics**
(file names containing `MOSAIC`). Only the NISP Y-band `BGSUB-MOSAIC` tile
has been tested; the other bands and the `RMS` / `FLAG` mosaics have the
same layout but were not tried. Euclid catalogues and single-exposure
frames are not images in the primary HDU and are out of scope for now.
## How Python and Rust fit together (PyO3)

If you are an astronomer who knows Python but has never touched Rust, this
is the one thing to understand: **you never write or run Rust yourself.**
You install the package and `import` it like `numpy` or `astropy`.

Under the hood there are three pieces:

| Piece | Where | What it does |
|---|---|---|
| The Rust code | `euclid_fast/src/lib.rs` | Does the actual work: memory-maps the FITS file, runs the parallel computation, converts sky coordinates to pixels |
| **PyO3** | a dependency in `euclid_fast/Cargo.toml` | A Rust library that turns Rust functions and structs into Python functions and objects |
| **maturin** | `build-backend` in `euclid_fast/pyproject.toml` | Compiles the Rust into a shared library and installs it in your Python environment (`pip` calls it for you) |

PyO3 shows up in `lib.rs` as a handful of annotations. Here is the whole
mapping, taken from the real code:

```rust
#[pyfunction]                                   // -> a Python function
fn fits_stats(path: &str) -> PyResult<FitsStats> { /* ... */ }

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
- **A Rust toolchain, version 1.85 or newer** (the crate uses Rust edition
  2024; developed with 1.100). Install it with [rustup](https://rustup.rs/):
  - Linux / macOS: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  - Windows: download `rustup-init.exe` from <https://rustup.rs/>. It will
    ask you to install the Visual Studio C++ Build Tools; accept.
  - Then **open a new terminal** and check with `rustc --version`.
- **A C linker**, which is normally already there (`build-essential` on
  Debian/Ubuntu, Xcode command-line tools on macOS, the Build Tools on
  Windows).

Everything else (maturin, PyO3 and the other Rust libraries) is downloaded
automatically during the build, so the first install needs internet access.

> Tested on Debian 13 (x86_64) with Python 3.13 and Rust 1.100. macOS and
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
[What kind of files does it work with?](#what-kind-of-files-does-it-work-with):
an uncompressed 2-D image in the primary HDU, with an `RA---TAN` WCS if you
want cutouts. Euclid MER mosaics are of this kind; see
[the benchmark section](#run-the-benchmark-on-your-own-fits-file) for where
to get one.

```python
import numpy as np
import euclid_fast

path = "tile.fits"

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

Reference:

| Function | Returns | Fields |
|---|---|---|
| `fits_stats(path)` | `FitsStats` | `min`, `max`, `mean`, `naxis1`, `naxis2`, `bitpix`, `threads_used`, `elapsed_ms` |
| `extract_cutout(path, ra_deg, dec_deg, width, height)` | `Cutout` | `data` (flat list of floats, row-major), `width`, `height`, `pixel_x_center`, `pixel_y_center`, `elapsed_ms` |
| `export_cutout_parquet(path, ra_deg, dec_deg, width, height, out_path)` | `ParquetExport` | `out_path`, `rows`, `elapsed_ms` |
| `detected_cpus()` | `int` | number of logical cores found on this machine |

Notes: `Cutout.data` is a plain Python list, so convert it with
`np.array(...)` as above (fine for cutouts; it is not meant for
multi-megapixel regions). The Parquet file has one row per pixel with
columns `x`, `y` (absolute pixel coordinates in the source image) and
`value`. All errors (file not found, unsupported `BITPIX`, cutout outside
the image, no WCS in the header...) are raised as `ValueError`.

## Run the tests (optional)

`tests/test_input_validation.py` builds small FITS files with astropy and
checks that supported files work and unsupported ones are rejected with a
clear error. With `euclid_fast` installed and `pip install -r
scripts/requirements.txt` done:

```bash
python tests/test_input_validation.py     # prints "all input-validation checks passed"
```

(`pytest tests/` works too.)

## Run the benchmark on your own FITS file

The script `scripts/benchmark_vs_astropy_real.py` times `euclid_fast`
against `astropy` on a file of your choice, and checks that both give the
same answer.

**1. Get a FITS file.** Use any file that matches the
[requirements above](#what-kind-of-files-does-it-work-with). To reproduce the numbers in this README, use
the Euclid Q1 tile they were measured on, from the public IRSA mirror
(anonymous S3 bucket `nasa-irsa-euclid-q1`, no account needed, 1.47 GB):

<https://nasa-irsa-euclid-q1.s3.amazonaws.com/q1/MER/102160339/NISP/EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE102160339-3A9BC7_20241024T231706.521789Z_00.00.fits>

Download it with your browser, `curl` or `wget`, and put it anywhere.

**2. Install the benchmark's two dependencies** (inside the same virtual
environment you installed `euclid_fast` in):

```bash
pip install -r scripts/requirements.txt     # numpy + astropy
```

**3. Run it**, giving the path to the file:

```bash
python scripts/benchmark_vs_astropy_real.py /path/to/tile.fits
```

Optional flags: `--ra` / `--dec` (cutout center in degrees; defaults to the
image center from the FITS header), `--size` (cutout width and height in
pixels, default 50) and `--runs` (repetitions per measurement, default 5).

**4. Read the output.** It prints, for each of the two operations, the median
time of both tools, the speedup, and a correctness check:

```
--- 1. Global stats (min/max/mean over the whole file) ---
astropy (memmap, numpy)      :   476.15 ms
euclid_fast (Rust/rayon)     :   130.28 ms
Speedup                      : ×3.65  (72.6% less time)
Result check: astropy min/max/mean = (-111.43283081054688, 15220.939453125, 1.0181564092636108)
              rust    min/max/mean = (-111.43283081054688, 15220.939453125, 1.0181568676826722)

--- 2. 50x50 cutout extraction by RA/Dec (273.071660, 68.000000) ---
astropy (WCS + memmap)       :    2.663 ms
euclid_fast (Rust/mmap)      :    0.216 ms
Speedup                      : ×12.36  (91.9% less time)
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
| Build error mentioning `edition2024` | Your Rust is older than 1.85. Run `rustup update`. |
| `linker 'cc' not found` (Linux) / `link.exe not found` (Windows) | Install a C toolchain: `sudo apt install build-essential`, or the Visual Studio C++ Build Tools. |
| `ModuleNotFoundError: No module named 'euclid_fast'` | You are in a different Python than the one you installed into. Activate the virtual environment (`source .venv/bin/activate`) and retry; check with `pip show euclid_fast`. |
| `ValueError: the primary HDU holds no image (NAXIS = 0)`, `NAXIS = 3 is not supported`, `BITPIX ... is not supported` or `no END card found` | The file is not a plain uncompressed 2-D image in the primary HDU (compressed, multi-extension, cube, table, not FITS...). See [What kind of files does it work with?](#what-kind-of-files-does-it-work-with). |
| `ValueError: unsupported WCS projection ...` | The cutout functions only implement the plain `RA---TAN` / `DEC--TAN` projection. SIN, SIP and other WCS types are not supported yet. |
| `ValueError: this file has no WCS header` | The cutout functions need `CRVAL1/2`, `CRPIX1/2` and the `CD` matrix. `fits_stats` still works without them. |
| `ValueError: cutout outside the image` | The RA/Dec you asked for falls outside the image. Check the coordinates against the header's `CRVAL1/2`. |
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
auto-sizes itself to the host's logical core count — there is no hardcoded
thread count, so the same binary scales itself from a laptop to a
many-core workstation.

## What it does today

Three operations, each independently benchmarked against `astropy` on
identical input and compared for numerical correctness, not just speed:

| Function | What it computes | Notes |
|---|---|---|
| `fits_stats(path)` | Global `min` / `max` / `mean` over the primary HDU | `mmap` + `rayon`, chunked reduction tuned for this workload |
| `extract_cutout(path, ra_deg, dec_deg, width, height)` | Pixel-space cutout around a sky position | Standard TAN (gnomonic) WCS, FITS 1-indexed convention |
| `export_cutout_parquet(path, ra_deg, dec_deg, width, height, out_path)` | Same cutout, written as Parquet | One row per pixel: columns `x`, `y` (absolute pixel coords in the source image), `value` (physical value, `BZERO`/`BSCALE` applied) |

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
Core i7-8559U (4 cores / 8 threads), 31 GB RAM, no discrete GPU —
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

- **No compressed-tile support yet.** Only uncompressed FITS data is
  handled. Rice/GZIP-compressed tiles (common in real archival data)
  require a decompression step that `euclid_fast` does not implement
  yet; that step necessarily costs a copy, but should still outperform
  the `cfitsio`/GIL path once added.
- **No SIP distortion.** The WCS transform is a pure linear TAN
  projection (`CRVAL`/`CRPIX`/`CD` matrix only). Real survey data with
  significant optical distortion (SIP polynomial terms) will need that
  added before cutout positions can be trusted at the sub-pixel level.
- **Single HDU, single file.** No multi-extension FITS, no batch
  processing across a file list (yet — see Roadmap).

- **Real-data validation is currently one file, one band.** The results
  above come from a single Q1 NISP tile out of the ~35 TB Q1 release —
  enough to confirm the approach works unmodified on a real archival
  product, not yet a representative sample across instruments, bands and
  tiles.
- **Tested on one platform.** Linux (Debian 13, x86_64) only so far.

## Roadmap

This is the first of four planned components for Euclid-scale data
processing, chosen first specifically because it needed no GPU/cluster
and is the foundation the others build on:

1. ✅ **FITS streaming / cutout / Parquet export** (this package).
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