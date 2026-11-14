## Machines

| Label | Processor | Threads | RAM | System | GPU |
|---|---|---|---|---|---|
| i7 (NUC) | Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz | 8 | 31.2 GiB | Linux 6.12.115+deb13.1-amd64 (Python 3.13.5) | none used |
| i9 (PC) | Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz | 20 | 63.8 GiB | Windows 11 (Python 3.12.10) | RTX 4070 Ti |

## Time to compute (milliseconds, median, lower is better)

| Operation | astropy CPU i7 | astropy CPU i9 | EuclidRust CPU i7 NUC | EuclidRust CPU i9 | Euclid GPU RTX 4070 Ti |
|---|---|---|---|---|---|
| **raw tile (1.47 GB)** | | | | | |
| min / max / mean of the whole image | 478 | 1,187 | 132 | 119 | 194 |
| same, image already on the GPU (copied once, 214 ms) |  |  |  |  | 10.1 |
| 50×50 cutout by RA/Dec | 2.70 | 3.33 | 0.22 | 0.23 | (runs on the CPU) |
| **Rice-compressed tile (0.31 GB, tile-compressed)** | | | | | |
| min / max / mean of the whole image | 3,603 | 3,849 | 1,211 | 384 | 67.3 |
| 50×50 cutout by RA/Dec | 17.1 | 19.3 | 3.55 | 2.51 | (runs on the CPU) |

## How many times faster than astropy on the same machine

| Operation | EuclidRust CPU i7 NUC | EuclidRust CPU i9 | Euclid GPU RTX 4070 Ti |
|---|---|---|---|
| raw tile: min / max / mean of the whole image | ×3.6 | ×10.0 | ×6.1 |
| raw tile: 50×50 cutout by RA/Dec | ×12.1 | ×14.6 | n/a |
| Rice-compressed tile: min / max / mean of the whole image | ×3.0 | ×10.0 | ×57.2 |
| Rice-compressed tile: 50×50 cutout by RA/Dec | ×4.8 | ×7.7 | n/a |
## How to read these numbers

Same file, same Python/numpy/astropy versions, data already in the OS cache, median of several runs, identical min/max/mean on every machine.

- **astropy is slower on the i9 than on the i7:** astropy/numpy use one core, so the extra cores do not help. FITS data is big-endian and numpy swaps the bytes of every value: numpy's `min()` on 64 MB of float32 runs at 20.3 GB/s (PC) and 23.8 GB/s (NUC) on ordinary data, but at 6.9 GB/s (PC, Windows) and 15.0 GB/s (NUC, Linux) on big-endian data. That column mostly measures numpy's big-endian path on Windows, not the processor.
- **EuclidRust gains little on the i9 over the i7 for the raw tile (119 ms vs 132 ms, ×1.1)** despite more than twice the cores and a third more clock speed: the computation is bound by memory bandwidth, not by the processor. The PC's memory runs at 2933 MT/s (measured at about 37 GB/s with many threads), the NUC's at 2400 MT/s. On the Rice-compressed tile, decoding each tile leaves more work per byte fetched, so the i9 pulls further ahead (384 ms vs 1,211 ms, ×3.2).
- **The GPU is slower than the CPU on the raw image:** the 1.47 GB cross PCIe 3.0 (12.9 GB/s measured, 114 ms at best) after a copy into page-locked memory, while the CPU now reads them in 119 ms. The computation on the card takes 10 ms. The GPU wins when the transfer is small (Rice-compressed file, 0.31 GB) or paid once (image already in GPU memory).

