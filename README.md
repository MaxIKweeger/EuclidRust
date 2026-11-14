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

All three operate on the primary HDU of a single-extension FITS file with
standard integer or floating-point `BITPIX` (`8`, `16`, `32`, `-32`,
`-64`) and, for the cutout functions, a linear WCS defined by
`CRVAL1/2`, `CRPIX1/2`, `CD1_1/CD1_2/CD2_1/CD2_2` with `CTYPE1/2` of
`RA---TAN` / `DEC--TAN`.

## Correctness: same results, not just faster ones

Every benchmark below compares `euclid_fast` against `astropy.io.fits` /
`astropy.wcs.WCS` on the **same file**. `min` and `max` are bit-for-bit
identical between the two implementations, `mean` agrees to 7 significant
figures (see below), and pixel values inside the cutout match exactly. The
speedups are not the result of computing something different, less, or
approximate; they come from removing interpreter and copy overhead around
the same arithmetic.

## Benchmarks

All numbers are measured on a **real Euclid Quick Data Release 1 (Q1)
product** pulled from the public IRSA mirror (`nasa-irsa-euclid-q1` S3
bucket, anonymous access, no credentials needed):

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
comparable state for both tools. The benchmark script is
[`scripts/benchmark_vs_astropy_real.py`](scripts/benchmark_vs_astropy_real.py).

| Operation | `astropy` | `euclid_fast` | Speedup | Time saved |
|---|---|---|---|---|
| Global stats (min/max/mean, full 1.47 GB) | 478.30 ms | 130.35 ms | ×3.67 | −72.7% |
| Cutout extraction (50×50 px, by RA/Dec) | 2.664 ms | 0.219 ms | ×12.16 | −91.8% |

`min` and `max` matched bit-for-bit between the two implementations.
`mean` matched to 7 significant figures (1.0181564... vs 1.0181568...) —
the residual difference is floating-point summation order (NumPy's
pairwise summation vs. this project's parallel tree reduction;
floating-point addition is not strictly associative), not a computational
error.

Two things are worth noting about *why* the gap differs between the two
operations:

- The **cutout** benchmark shows the larger relative gain, because it is
  dominated by per-call Python/WCS/NumPy object overhead — exactly the
  overhead this project is designed to remove. This is also the most
  common real-world access pattern (zooming in on one object), so it's
  the number that matters most in practice.
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