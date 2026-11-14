"""
Regenerates euclidrust/src/kernels.ptx, the GPU kernels compiled to PTX, from
euclidrust/src/kernels.cu.

Why: the library ships these kernels inside the binary, so that a machine only needs the
NVIDIA *driver* to use its GPU (not the CUDA toolkit). The driver turns the PTX into code
for the card at hand. Run this script after editing kernels.cu, then rebuild the library.

It needs a CUDA toolkit (for NVRTC) on the machine where it runs, and a euclidrust that
can compile the kernels: the very first build, or one whose kernels.ptx is stale, falls
back to NVRTC by itself, so the order is always:

    cd euclidrust && maturin develop --release     # works even with an empty kernels.ptx
    python ../scripts/build_ptx.py                  # writes src/kernels.ptx
    maturin develop --release                       # embeds it
"""
import sys
from pathlib import Path

try:
    import euclidrust
except ImportError:
    sys.exit("euclidrust is not installed: build it first (see the docstring).")

target = Path(__file__).resolve().parent.parent / "euclidrust" / "src" / "kernels.ptx"
try:
    ptx = euclidrust._compile_kernels_ptx()
except ValueError as e:
    sys.exit(f"cannot compile the kernels: {e}")
target.write_text(ptx.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
print(f"wrote {target} ({len(ptx):,} characters)")
