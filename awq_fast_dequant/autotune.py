"""
Shape-keyed block size autotuning for the v3 kernel.

On first encountering a given tensor shape, runs a timing sweep across
BLOCK_SIZE_CANDIDATES and caches the fastest one. Subsequent calls for the
same shape reuse the cached decision. The cache persists to a per-user
cache directory, not a path relative to the current working directory, so
behavior is consistent regardless of where the caller's process runs from.
"""

import json
import os
import statistics
import time

import torch

from ._build import _load_v3

BLOCK_SIZE_CANDIDATES = [64, 128, 256, 512, 1024]
NUM_AUTOTUNE_WARMUP = 3
NUM_AUTOTUNE_TIMED_RUNS = 10

_block_size_cache = {}


def _cache_path() -> str:
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = os.path.join(cache_home, "awq_fast_dequant")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "block_size_cache.json")


def _load_cache_from_disk() -> dict:
    path = _cache_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {tuple(int(x) for x in key.split(",")): value for key, value in raw.items()}


def _save_cache_to_disk() -> None:
    path = _cache_path()
    serializable = {",".join(str(x) for x in key): value for key, value in _block_size_cache.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


_block_size_cache.update(_load_cache_from_disk())


def _shape_key(qweight: torch.Tensor, scales: torch.Tensor) -> tuple:
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = K // scales.shape[0]
    return (K, M, group_size)


def _timed_call(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def _autotune_block_size(qweight, scales, zeros) -> int:
    v3 = _load_v3()
    means = {}
    for block_size in BLOCK_SIZE_CANDIDATES:
        fn = lambda bs=block_size: v3.awq_dequantize_v3(qweight, scales, zeros, bs)
        for _ in range(NUM_AUTOTUNE_WARMUP):
            fn()
        latencies = [_timed_call(fn) for _ in range(NUM_AUTOTUNE_TIMED_RUNS)]
        means[block_size] = statistics.mean(latencies)
    return min(means, key=means.get)


def smart_dequantize(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """
    Dequantize AWQ 4-bit weights, automatically selecting and caching the
    fastest block size for this tensor shape.

    The first call for a new shape pays the cost of a timing sweep across
    BLOCK_SIZE_CANDIDATES. Every subsequent call for the same shape reuses
    the cached decision, including across process restarts.
    """
    key = _shape_key(qweight, scales)

    if key not in _block_size_cache:
        _block_size_cache[key] = _autotune_block_size(qweight, scales, zeros)
        _save_cache_to_disk()

    v3 = _load_v3()
    return v3.awq_dequantize_v3(qweight, scales, zeros, _block_size_cache[key])


def clear_cache() -> None:
    """Clear the in-memory and on-disk autotuning cache."""
    _block_size_cache.clear()
    path = _cache_path()
    if os.path.exists(path):
        os.remove(path)