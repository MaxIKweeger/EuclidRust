"""
Comparaison chronometree : astropy (standard astro) vs euclid_fast (Rust),
sur le meme fichier et les memes operations, sur ce NUC.

Operations comparees :
  1. Stats globales (min/max/mean) sur le fichier entier (1 Go).
  2. Extraction d'un cutout 50x50 par coordonnees celestes (RA/Dec).

Chaque mesure est repetee plusieurs fois (apres un run de chauffe pour que
le cache disque OS soit dans le meme etat des deux cotes) ; on garde la
mediane pour eviter qu'un pic isole fausse la comparaison.
"""
import statistics
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import euclid_fast

ROOT = Path(__file__).resolve().parent.parent
PATH = str(ROOT / "data" / "real" / "euclid_real_Y.fits")
RA, DEC = 273.0716597, 68.0
CUTOUT_SIZE = 50
N_RUNS = 5


def median_ms(fn):
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.median(times), times


# --- Chauffe : un passage de chaque methode pour que le fichier soit en cache OS ---
with fits.open(PATH, memmap=True) as hdul:
    _ = hdul[0].data.min()
_ = euclid_fast.fits_stats(PATH)


# --- 1. Stats globales ---
def astropy_stats():
    with fits.open(PATH, memmap=True) as hdul:
        data = hdul[0].data
        return float(data.min()), float(data.max()), float(data.mean())


def rust_stats():
    s = euclid_fast.fits_stats(PATH)
    return s.min, s.max, s.mean


astropy_stats_result = astropy_stats()
rust_stats_result = rust_stats()

t_astropy_stats, _ = median_ms(astropy_stats)
t_rust_stats, _ = median_ms(rust_stats)


# --- 2. Cutout par RA/Dec ---
def astropy_cutout():
    with fits.open(PATH, memmap=True) as hdul:
        header = hdul[0].header
        wcs = WCS(header)
        x, y = wcs.all_world2pix(RA, DEC, 0)  # 0-indexe
        x0 = int(round(float(x))) - CUTOUT_SIZE // 2
        y0 = int(round(float(y))) - CUTOUT_SIZE // 2
        cutout = hdul[0].data[y0 : y0 + CUTOUT_SIZE, x0 : x0 + CUTOUT_SIZE]
        return np.array(cutout)  # force la lecture reelle


def rust_cutout():
    c = euclid_fast.extract_cutout(PATH, RA, DEC, CUTOUT_SIZE, CUTOUT_SIZE)
    return c.data


astropy_cutout_result = astropy_cutout()
rust_cutout_result = rust_cutout()

t_astropy_cutout, _ = median_ms(astropy_cutout)
t_rust_cutout, _ = median_ms(rust_cutout)


def pct_faster(t_baseline, t_new):
    """% de temps gagne par rapport a la baseline (positif = plus rapide)."""
    return (1 - t_new / t_baseline) * 100.0


def speedup(t_baseline, t_new):
    return t_baseline / t_new


print("=" * 70)
print(f"Fichier : {PATH}  (1,03 Go)")
print(f"Médiane sur {N_RUNS} runs, après chauffe cache OS")
print("=" * 70)

print("\n--- 1. Stats globales (min/max/mean sur tout le fichier) ---")
print(f"astropy (memmap, numpy)      : {t_astropy_stats:8.2f} ms")
print(f"euclid_fast (Rust/rayon)     : {t_rust_stats:8.2f} ms")
print(f"Gain                         : ×{speedup(t_astropy_stats, t_rust_stats):.2f}  "
      f"({pct_faster(t_astropy_stats, t_rust_stats):.1f}% de temps en moins)")
print(f"Cohérence résultats : astropy min/max/mean = {astropy_stats_result}")
print(f"                      rust    min/max/mean = {rust_stats_result}")

print("\n--- 2. Extraction cutout 50x50 par RA/Dec ---")
print(f"astropy (WCS + memmap)       : {t_astropy_cutout:8.3f} ms")
print(f"euclid_fast (Rust/mmap)      : {t_rust_cutout:8.3f} ms")
print(f"Gain                         : ×{speedup(t_astropy_cutout, t_rust_cutout):.2f}  "
      f"({pct_faster(t_astropy_cutout, t_rust_cutout):.1f}% de temps en moins)")
print(f"Cohérence résultats : astropy min/max = {astropy_cutout_result.min()}/{astropy_cutout_result.max()}")
print(f"                      rust    min/max = {min(rust_cutout_result)}/{max(rust_cutout_result)}")
