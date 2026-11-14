# euclid_fast

High-performance FITS I/O for Python, written in Rust and exposed with
[PyO3](https://pyo3.rs/). This directory is the Rust crate and Python
package; the full documentation lives in the
[repository README](../README.md).

```bash
pip install .        # run from this directory; needs a Rust toolchain (>= 1.88)
```

```python
import euclid_fast

stats = euclid_fast.fits_stats("tile.fits")                    # min / max / mean
cutout = euclid_fast.extract_cutout("tile.fits", 83.82, -5.39, 50, 50)  # RA/Dec cutout
euclid_fast.export_cutout_parquet("tile.fits", 83.82, -5.39, 50, 50, "cutout.parquet")
```

See the [repository README](../README.md) for installation details, usage,
how to run the benchmark, benchmark results and known limitations.