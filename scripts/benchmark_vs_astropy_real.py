"""
Timed comparison: astropy (the astronomy standard) vs euclidrust (Rust),
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

The FITS file must contain a 2-D image (raw, or tile-compressed with RICE_1,
GZIP_1 or GZIP_2; a compressed image is in extension 1, so pass --hdu 1) with a standard
RA---TAN / DEC--TAN (or RA---TPV / DEC--TPV) WCS. For a Euclid MER mosaic
tile the image is in the primary HDU. For calibrated frames (VIS, NISP) the
images live in extensions: choose one with --hdu (an EXTNAME or a 0-based
index), for example:

    python scripts/benchmark_vs_astropy_real.py frame.fits --hdu DET11.SCI

By default the cutout is centered on the sky position of the image center,
computed from the file's own WCS, so any such file works without choosing
coordinates. If the file has no sky WCS (for example raw detector frames),
only the global-stats comparison is run.

Extra options:
  --gpu          also time the NVIDIA GPU (needs a build with the `cuda` feature):
                 end to end, and with the image already in the card's memory.
  --json FILE    write every measurement, with a description of the machine, to FILE
                 (scripts/make_comparison_table.py merges such files from several
                 machines into one table).
  --label TEXT   a name for this machine in that file, e.g. "i9 (PC)".
"""
import argparse
import json
import os
import platform
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
    import euclidrust
except ImportError:
    sys.exit("The euclidrust module is not installed in this Python environment.\n"
             "Build and install it first, from the project root:\n"
             "    pip install ./euclidrust\n"
             "(see the README, section 'Install').")



def machine_info():
    """Processor, memory and operating system of this machine, for the JSON report."""
    info = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu": None, "cpu_threads": os.cpu_count(), "ram_gib": None}
    try:
        if platform.system() == "Windows":
            import ctypes
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                            ("avail_phys", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                            ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                            ("avail_virtual", ctypes.c_ulonglong), ("avail_ext", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            info["ram_gib"] = round(status.total_phys / 2**30, 1)
        else:
            with open("/proc/cpuinfo") as f:
                info["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
            with open("/proc/meminfo") as f:
                info["ram_gib"] = round(next(int(line.split()[1]) for line in f if line.startswith("MemTotal")) / 2**20, 1)
    except Exception:
        pass
    return info


def cpu_stats(path, hdu):
    """euclidrust's statistics computed on the CPU. The default device is "auto", which may pick
    the GPU, so the CPU is asked for explicitly (older builds have no `device` argument and
    only know the CPU)."""
    try:
        return euclidrust.fits_stats(path, hdu=hdu, device="cpu")
    except TypeError:
        return euclidrust.fits_stats(path, hdu=hdu)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark euclidrust against astropy on a FITS image.")
    p.add_argument("file", type=Path, help="path to the FITS image to benchmark")
    p.add_argument("--hdu", type=lambda s: int(s) if s.isdigit() else s, default=None,
                   help="HDU holding the image: an EXTNAME or a 0-based index (default: the primary HDU)")
    p.add_argument("--ra", type=float, default=None,
                   help="cutout center RA in degrees (default: the image center)")
    p.add_argument("--dec", type=float, default=None,
                   help="cutout center Dec in degrees (default: the image center)")
    p.add_argument("--size", type=int, default=50, help="cutout width and height in pixels (default: 50)")
    p.add_argument("--runs", type=int, default=5, help="repetitions per measurement (default: 5)")
    p.add_argument("--gpu", action="store_true", help="also time the NVIDIA GPU (build with the cuda feature)")
    p.add_argument("--json", type=Path, default=None, help="write all the measurements to this JSON file")
    p.add_argument("--label", default=None, help="name of this machine in the JSON file, e.g. 'i9 (PC)'")
    p.add_argument("--workload", default=None, help="name of this workload in the JSON file, e.g. 'raw tile'")
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

    hdu = args.hdu
    hdu_idx = 0 if hdu is None else hdu          # astropy HDU selector
    with fits.open(path, memmap=True) as hdul:
        header = hdul[hdu_idx].header
        # astropy cannot memory-map images that need BZERO/BSCALE/BLANK scaling
        memmap = not any(k in header for k in ("BZERO", "BSCALE", "BLANK"))
        compressed = isinstance(hdul[hdu_idx], fits.CompImageHDU)
        # Default cutout center: an integer pixel near the middle of the image,
        # converted to sky coordinates with the file's own WCS.
        try:
            cx, cy = header["NAXIS1"] // 2 + 1, header["NAXIS2"] // 2 + 1      # FITS 1-indexed
            c_ra, c_dec = WCS(header).all_pix2world(cx, cy, 1)
            ra = args.ra if args.ra is not None else float(c_ra)
            dec = args.dec if args.dec is not None else float(c_dec)
        except Exception:
            ra, dec = args.ra, args.dec
    size, n_runs = args.size, args.runs

    # --- Warm-up: one pass of each method so the file is in the OS cache ---
    with fits.open(path, memmap=memmap) as hdul:
        _ = hdul[hdu_idx].data.min()
    _ = cpu_stats(path, hdu)

    # --- 1. Global stats ---
    def astropy_stats():
        with fits.open(path, memmap=memmap) as hdul:
            data = hdul[hdu_idx].data
            lo, hi, mean = float(data.min()), float(data.max()), float(data.mean())
            if mean != mean:   # NaN pixels: numpy's plain functions give NaN, so use the NaN-aware ones
                lo, hi, mean = float(np.nanmin(data)), float(np.nanmax(data)), float(np.nanmean(data, dtype=np.float64))
            return lo, hi, mean

    def rust_stats():
        s = cpu_stats(path, hdu)
        return s.min, s.max, s.mean

    astropy_stats_result = astropy_stats()
    rust_stats_result = rust_stats()
    n_nan = cpu_stats(path, hdu).n_nan
    t_astropy_stats, _ = median_ms(astropy_stats, n_runs)
    t_rust_stats, _ = median_ms(rust_stats, n_runs)

    gpu = None
    if args.gpu:
        gpu = {"available": False}
        try:
            info = euclidrust.gpu_info()
            gpu.update(available=bool(info.available), name=info.name, memory_mb=info.memory_total_mb)
            if not info.available:
                gpu["reason"] = info.message
        except AttributeError:
            gpu["reason"] = "this euclidrust build has no GPU support"
        if gpu["available"]:
            try:
                cpu_ref = cpu_stats(path, hdu)
                euclidrust.fits_stats(path, hdu=hdu, device="gpu")          # warm-up: CUDA start-up
                gpu_stats = euclidrust.fits_stats(path, hdu=hdu, device="gpu")
                gpu["end_to_end_ms"], _ = median_ms(lambda: euclidrust.fits_stats(path, hdu=hdu, device="gpu"), n_runs)
                gpu["matches_cpu"] = bool((gpu_stats.min, gpu_stats.max) == (cpu_ref.min, cpu_ref.max)
                                          and abs(gpu_stats.mean - cpu_ref.mean) <= 1e-9 * max(1.0, abs(cpu_ref.mean)))
                resident = euclidrust.GpuImage(path, hdu=hdu)
                gpu["upload_ms"] = resident.upload_ms
                gpu["resident_ms"], _ = median_ms(resident.stats, n_runs)
                gpu["matches_cpu"] = bool((gpu_stats.min, gpu_stats.max) == (cpu_ref.min, cpu_ref.max)
                                          and abs(gpu_stats.mean - cpu_ref.mean) <= 1e-9 * max(1.0, abs(cpu_ref.mean)))
                del resident
            except ValueError as e:        # this kind of image is not handled on the GPU (yet)
                gpu["unsupported"] = str(e)
    report = {
        "machine": {"label": args.label, **machine_info()},
        "workload": args.workload,
        "file": {"name": os.path.basename(path), "size_bytes": os.path.getsize(path), "hdu": hdu, "compressed": compressed},
        "runs": n_runs,
        "euclidrust_threads": euclidrust.detected_cpus(),
        "stats": {"astropy_ms": t_astropy_stats, "euclidrust_cpu_ms": t_rust_stats,
                  "astropy_result": list(astropy_stats_result), "euclidrust_result": list(rust_stats_result), "gpu": gpu},
        "cutout": None,
    }

    def write_report():
        if args.json:
            args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nmeasurements written to {args.json}")

    # --- 2. Cutout by RA/Dec ---
    def astropy_cutout():
        with fits.open(path, memmap=memmap) as hdul:
            wcs = WCS(hdul[hdu_idx].header)
            x, y = wcs.all_world2pix(ra, dec, 0)  # 0-indexed
            x0 = int(round(float(x))) - size // 2
            y0 = int(round(float(y))) - size // 2
            if compressed:   # astropy's way of reading a region without decompressing everything
                cutout = hdul[hdu_idx].section[y0 : y0 + size, x0 : x0 + size]
            else:
                cutout = hdul[hdu_idx].data[y0 : y0 + size, x0 : x0 + size]
            return np.array(cutout)  # force the actual read

    def rust_cutout():
        return euclidrust.extract_cutout(path, ra, dec, size, size, hdu=hdu).data

    cutout_skipped = None
    if ra is None or dec is None or not np.isfinite([ra, dec]).all():
        cutout_skipped = "no usable sky WCS in this HDU (pass --ra and --dec to force a position)"
    else:
        try:
            astropy_cutout_result = astropy_cutout()
            rust_cutout_result = rust_cutout()
            t_astropy_cutout, _ = median_ms(astropy_cutout, n_runs)
            t_rust_cutout, _ = median_ms(rust_cutout, n_runs)
            same_pixels = np.array_equal(
                astropy_cutout_result.astype(np.float64).ravel(), np.asarray(rust_cutout_result), equal_nan=True
            )
            report["cutout"] = {"size": size, "astropy_ms": t_astropy_cutout, "euclidrust_cpu_ms": t_rust_cutout,
                                "identical_pixels": bool(same_pixels)}
        except ValueError as e:        # euclidrust refused the file/position: say why
            cutout_skipped = str(e)

    print("=" * 70)
    print(f"File: {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")
    print(f"HDU: {'primary' if hdu is None else repr(hdu)}"
          + ("" if memmap else "   (BZERO/BSCALE present: astropy cannot memory-map, it reads the data into RAM)")
          + ("   (tile-compressed: astropy decompresses the whole image for the global stats)" if compressed else ""))
    if n_nan:
        print(f"{n_nan:,} NaN pixels: both sides leave them out (astropy side: numpy's nanmin / nanmax / nanmean)")
    print(f"Median over {n_runs} runs, after OS cache warm-up")
    print(f"CPU threads used by euclidrust: {euclidrust.detected_cpus()}")
    print("=" * 70)

    print("\n--- 1. Global stats (min/max/mean over the whole file) ---")
    print(f"astropy (memmap, numpy)      : {t_astropy_stats:8.2f} ms")
    print(f"euclidrust CPU (Rust/rayon)  : {t_rust_stats:8.2f} ms")
    print(f"Speedup                      : ×{speedup(t_astropy_stats, t_rust_stats):.2f}  "
          f"({pct_faster(t_astropy_stats, t_rust_stats):.1f}% less time)")
    print(f"Result check: astropy min/max/mean = {astropy_stats_result}")
    print(f"              rust    min/max/mean = {rust_stats_result}")
    if gpu is not None:
        if not gpu["available"]:
            print(f"GPU                          : not available ({gpu.get('reason', '')})")
        elif "end_to_end_ms" not in gpu:
            print(f"GPU ({gpu['name']})  : {gpu.get('unsupported', 'not measured')}")
        else:
            print(f"GPU {gpu['name']:<24} : {gpu['end_to_end_ms']:8.2f} ms end to end"
                  f" (same results as the CPU: {gpu['matches_cpu']})")
            if "resident_ms" in gpu:
                print(f"GPU, image already on the card: {gpu['resident_ms']:8.2f} ms"
                      f"  (the copy to the card took {gpu['upload_ms']:.0f} ms, once)")

    if cutout_skipped:
        print(f"\n--- 2. cutout comparison skipped ---\n{cutout_skipped}")
        write_report()
        return

    print(f"\n--- 2. {size}x{size} cutout extraction by RA/Dec ({ra:.6f}, {dec:.6f}) ---")
    print(f"astropy (WCS + memmap)       : {t_astropy_cutout:8.3f} ms")
    print(f"euclidrust (Rust/mmap)       : {t_rust_cutout:8.3f} ms")
    print(f"Speedup                      : ×{speedup(t_astropy_cutout, t_rust_cutout):.2f}  "
          f"({pct_faster(t_astropy_cutout, t_rust_cutout):.1f}% less time)")
    print(f"Result check: astropy min/max = {astropy_cutout_result.min()}/{astropy_cutout_result.max()}")
    print(f"              rust    min/max = {min(rust_cutout_result)}/{max(rust_cutout_result)}")
    print(f"              all {size * size} pixels identical: {same_pixels}")
    write_report()


if __name__ == "__main__":
    main()