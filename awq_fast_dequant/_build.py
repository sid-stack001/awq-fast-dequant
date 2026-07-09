"""
Lazy, cached loading of the compiled CUDA extensions.

Kernels are JIT-compiled on first use via torch's C++ extension loader,
against whatever PyTorch and CUDA toolchain is present in the caller's
environment. This trades a slower first call for portability: the kernel
does not need to be rebuilt for every possible PyTorch/CUDA combination
ahead of time, unlike a prebuilt wheel.
"""

import os
from functools import lru_cache

from torch.utils.cpp_extension import load

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kernels")


@lru_cache(maxsize=None)
def _load_v2():
    return load(
        name="awq_dequantize_v2",
        sources=[os.path.join(_KERNEL_DIR, "awq_dequantize_v2.cu")],
        verbose=False,
    )


@lru_cache(maxsize=None)
def _load_v3():
    return load(
        name="awq_dequantize_v3",
        sources=[os.path.join(_KERNEL_DIR, "awq_dequant_v3.cu")],
        verbose=False,
    )