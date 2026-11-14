"""
Timed comparison: astropy (the astronomy standard) vs euclid_fast (Rust),
on the same file and the same operations.

Operations compared:
  1. Global stats (min/max/mean) over the whole image.
  2. Extraction of a small cutout (default 50x50 px) by sky coordinates
     (RA/Dec).

Each measurement is repeated several times (after a warm-up run so that the
OS disk cache is in the same state on both sides); the median is kept so
that an isolated spike does not skew the comparison.

Usage (see the README for the full walkthrough):

    python scripts/benchmark_vs_astropy_real.py path/to/image.fits

The FITS file must be an uncompressed single-HDU image with a standard
RA---TAN / DEC--TAN WCS (for example a Euclid MER mosaic tile). By default
the cutout is centered on the image center (CRVAL1/CRVAL2 from the FITS
header), so any such file works without choosing coordinates.
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

try:
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
except ImportError:
    sys.exit("numpy and astropy are required for this benchmark:\n"
             "    pip install -r scripts/requirements.txt")

try:
    import euclid_fast
except ImportError:
    sys.exit("The euclid_fast module is not installed in this Python environment.\n"
             "Build and install it first, from the project root:\n"
             "    pip install ./euclid_fast\n"
             "(see the README, section 'Install').")



def parse_args():
    p = argparse.ArgumentParser(description="Benchmark euclid_fast against astropy on a FITS image.")
    p.add_argument("file", type=Path, help="path to the FITS image to benchmark")
    p.add_argument("--ra", type=float, default=None,
                   help="cutout center RA in degrees (default: CRVAL1 of the file)")
    p.add_argument("--dec", type=float, default=None,
                   help="cutout center Dec in degrees (default: CRVAL2 of the file)")
    p.add_argument("--size", type=int, default=50, help="cutout width and height in pixels (default: 50)")
    p.add_argument("--runs", type=int, default=5, help="repetitions per measurement (default: 5)")
    return p.parse_args()


def median_ms(fn, n_runs):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.median(times), times


def pct_faster(t_baseline, t_new):
    """% of time saved relative to the baseline (positive = faster)."""
    return (1 - t_new / t_baseline) * 100.0


def speedup(t_baseline, t_new):
    return t_baseline / t_new


def main():
    args = parse_args()
    path = str(args.file)
    if not args.file.exists():
        sys.exit(f"File not found: {args.file}")

    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        ra = args.ra if args.ra is not None else float(header["CRVAL1"])
        dec = args.dec if args.dec is not None else float(header["CRVAL2"])
    size, n_runs = args.size, args.runs

    # --- Warm-up: one pass of each method so the file is in the OS cache ---
    with fits.open(path, memmap=True) as hdul:
        _ = hdul[0].data.min()
    _ = euclid_fast.fits_stats(path)

    # --- 1. Global stats ---
    def astropy_stats():
        with fits.open(path, memmap=True) as hdul:
            data = hdul[0].data
            return float(data.min()), float(data.max()), float(data.mean())

    def rust_stats():
        s = euclid_fast.fits_stats(path)
        return s.min, s.max, s.mean

    astropy_stats_result = astropy_stats()
    rust_stats_result = rust_stats()
    t_astropy_stats, _ = median_ms(astropy_stats, n_runs)
    t_rust_stats, _ = median_ms(rust_stats, n_runs)

    # --- 2. Cutout by RA/Dec ---
    def astropy_cutout():
        with fits.open(path, memmap=True) as hdul:
            wcs = WCS(hdul[0].header)
            x, y = wcs.all_world2pix(ra, dec, 0)  # 0-indexed
            x0 = int(round(float(x))) - size // 2
            y0 = int(round(float(y))) - size // 2
            cutout = hdul[0].data[y0 : y0 + size, x0 : x0 + size]
            return np.array(cutout)  # force the actual read

    def rust_cutout():
        return euclid_fast.extract_cutout(path, ra, dec, size, size).data

    astropy_cutout_result = astropy_cutout()
    rust_cutout_result = rust_cutout()
    t_astropy_cutout, _ = median_ms(astropy_cutout, n_runs)
    t_rust_cutout, _ = median_ms(rust_cutout, n_runs)

    same_pixels = np.array_equal(
        astropy_cutout_result.astype(np.float64).ravel(), np.asarray(rust_cutout_result)
    )

    print("=" * 70)
    print(f"File: {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")
    print(f"Median over {n_runs} runs, after OS cache warm-up")
    print(f"CPU threads used by euclid_fast: {euclid_fast.detected_cpus()}")
    print("=" * 70)

    print("\n--- 1. Global stats (min/max/mean over the whole file) ---")
    print(f"astropy (memmap, numpy)      : {t_astropy_stats:8.2f} ms")
    print(f"euclid_fast (Rust/rayon)     : {t_rust_stats:8.2f} ms")
    print(f"Speedup                      : ×{speedup(t_astropy_stats, t_rust_stats):.2f}  "
          f"({pct_faster(t_astropy_stats, t_rust_stats):.1f}% less time)")
    print(f"Result check: astropy min/max/mean = {astropy_stats_result}")
    print(f"              rust    min/max/mean = {rust_stats_result}")

    print(f"\n--- 2. {size}x{size} cutout extraction by RA/Dec ({ra:.6f}, {dec:.6f}) ---")
    print(f"astropy (WCS + memmap)       : {t_astropy_cutout:8.3f} ms")
    print(f"euclid_fast (Rust/mmap)      : {t_rust_cutout:8.3f} ms")
    print(f"Speedup                      : ×{speedup(t_astropy_cutout, t_rust_cutout):.2f}  "
          f"({pct_faster(t_astropy_cutout, t_rust_cutout):.1f}% less time)")
    print(f"Result check: astropy min/max = {astropy_cutout_result.min()}/{astropy_cutout_result.max()}")
    print(f"              rust    min/max = {min(rust_cutout_result)}/{max(rust_cutout_result)}")
    print(f"              all {size * size} pixels identical: {same_pixels}")


if __name__ == "__main__":
    main()