"""
awq_fast_dequant

Custom CUDA kernels for AWQ 4-bit weight dequantization, with a wrapper
that automatically selects and caches the fastest execution configuration
per tensor shape.

Kernels are compiled on first use against the caller's installed PyTorch
and CUDA toolchain (see _build.py). Benchmarked on an RTX 3060 Laptop
(Ampere, 6GB VRAM); relative performance on other GPU architectures has
not been validated.
"""

from ._build import _load_v2, _load_v3
from .autotune import smart_dequantize, clear_cache

__all__ = ["dequantize_v2", "dequantize_v3", "smart_dequantize", "clear_cache"]
__version__ = "0.1.0"


def dequantize_v2(qweight, scales, zeros):
    """
    Dequantize AWQ 4-bit weights using the v2 kernel: one thread per
    packed 32-bit word, avoiding the redundant reads present in a naive
    one-thread-per-output-element implementation.
    """
    return _load_v2().awq_dequantize_v2(qweight, scales, zeros)


def dequantize_v3(qweight, scales, zeros, block_size: int = 128):
    """
    Dequantize AWQ 4-bit weights using the v3 kernel, with block size
    exposed as a caller-supplied parameter. See smart_dequantize() for
    automatic block size selection.
    """
    return _load_v3().awq_dequantize_v3(qweight, scales, zeros, block_size)