## Machines

| Label | Processor | Threads | RAM | System | GPU |
|---|---|---|---|---|---|
| i7 (NUC) | Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz | 8 | 31.2 GiB | Linux 6.12.115+deb13.1-amd64 (Python 3.13.5) | none used |
| i9 (PC) | Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz | 20 | 63.8 GiB | Windows 11 (Python 3.12.10) | RTX 4070 Ti |

## Time to compute (milliseconds, median, lower is better)

| Operation | astropy CPU i7 | astropy CPU i9 | EuclidRust CPU i7 NUC | EuclidRust CPU i9 | Euclid GPU RTX 4070 Ti |
|---|---|---|---|---|---|
| **raw tile (1.47 GB)** | | | | | |
| min / max / mean of the whole image | 478 | 1,299 | 132 | 125 | 243 |
| same, image already on the GPU (copied once, 265 ms) |  |  |  |  | 10.3 |
| 50×50 cutout by RA/Dec | 2.70 | 3.56 | 0.22 | 0.27 | (runs on the CPU) |
| **Rice-compressed tile (0.31 GB, tile-compressed)** | | | | | |
| min / max / mean of the whole image | 3,603 | 4,074 | 1,211 | 393 | 75.4 |
| 50×50 cutout by RA/Dec | 17.1 | 19.4 | 3.55 | 2.63 | (runs on the CPU) |

## How many times faster than astropy on the same machine

| Operation | EuclidRust CPU i7 NUC | EuclidRust CPU i9 | Euclid GPU RTX 4070 Ti |
|---|---|---|---|
| raw tile: min / max / mean of the whole image | ×3.6 | ×10.4 | ×5.3 |
| raw tile: 50×50 cutout by RA/Dec | ×12.1 | ×13.0 | n/a |
| Rice-compressed tile: min / max / mean of the whole image | ×3.0 | ×10.4 | ×54.0 |
| Rice-compressed tile: 50×50 cutout by RA/Dec | ×4.8 | ×7.4 | n/a |

