"""
Autotuning AWQ dequantization: picks the best block size per unique shape
by running a real timing sweep ONCE, then reusing that decision on every
subsequent call with the same shape.

Design:
    - shape_key = (K, M, group_size) -- the minimal set of values that
      determines which block size wins, based on our block_sweep.py findings.
    - First call for a new shape: pays the real cost of a timing sweep
    - Every subsequent call for that same shape: cache hit, near-zero
      overhead, launches directly with the remembered block size.
"""

import time
import statistics
import os
import json

import torch
from torch.utils.cpp_extension import load

BLOCK_SIZE_CANDIDATES = [64, 128, 256, 512, 1024] 
NUM_AUTOTUNE_WARMUP = 3
NUM_AUTOTUNE_TIMED_RUNS = 10

CACHE_DIR = "autotune_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "block_size_cache.json")

_block_size_cache = {}


def _load_cache_from_disk():
    
    if not os.path.exists(CACHE_FILE):
        return {}

    with open(CACHE_FILE, "r") as f:
        raw = json.load(f)

    # Convert "1536,8960,128" back into (1536, 8960, 128)
    return {
        tuple(int(x) for x in key_str.split(",")): value
        for key_str, value in raw.items()
    }


def _save_cache_to_disk():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Convert (1536, 8960, 128) into "1536,8960,128" for JSON compatibility
    serializable = {
        ",".join(str(x) for x in key): value
        for key, value in _block_size_cache.items()
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(serializable, f, indent=2)

_block_size_cache.update(_load_cache_from_disk())


def _shape_key(qweight: torch.Tensor, scales: torch.Tensor) -> tuple:
    """
    The minimal set of values that determines which block size is best
    for this shape. K and M determine total work; group_size affects the
    zeros/scales indexing pattern, which could plausibly shift which
    block size wins
    """
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = K // scales.shape[0]
    return (K, M, group_size)


def _timed_call(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000


def _autotune_block_size(qweight, scales, zeros, v3_kernel) -> int:
    """
    Runs a real timing sweep across BLOCK_SIZE_CANDIDATES for this exact
    shape, and returns whichever candidate had the lowest mean latency.
    This is the expensive path, only called once per unique shape.
    """
    print(f"  [autotune] New shape detected: {_shape_key(qweight, scales)}. "
          f"Running one-time block size sweep ...")

    candidate_means = {}
    for block_size in BLOCK_SIZE_CANDIDATES:
        fn = lambda: v3_kernel.awq_dequantize_v3(qweight, scales, zeros, block_size)

        for _ in range(NUM_AUTOTUNE_WARMUP):
            fn()

        latencies = [_timed_call(fn) for _ in range(NUM_AUTOTUNE_TIMED_RUNS)]
        candidate_means[block_size] = statistics.mean(latencies)
        print(f"    block_size={block_size}: {candidate_means[block_size]:.4f}ms")

    best_block_size = min(candidate_means, key=candidate_means.get)
    print(f"  [autotune] Chosen block_size={best_block_size} "
          f"({candidate_means[best_block_size]:.4f}ms)")
    return best_block_size


def smart_awq_dequantize(qweight, scales, zeros, v3_kernel) -> torch.Tensor:
    """
    The actual function real code would call. Transparently autotunes on
    first encounter with a new shape, reuses the cached decision after that.
    """
    key = _shape_key(qweight, scales)

    if key not in _block_size_cache:
        _block_size_cache[key] = _autotune_block_size(qweight, scales, zeros, v3_kernel)
        _save_cache_to_disk()  # persist immediately, don't wait for process exit

    block_size = _block_size_cache[key]
    return v3_kernel.awq_dequantize_v3(qweight, scales, zeros, block_size)


# ---------------------------------------------------------------------------
# Test / demonstration
# ---------------------------------------------------------------------------

GROUP_SIZE = 128
PACK_FACTOR = 8
DEVICE = "cuda"


def make_weights(input_size: int, output_size: int, seed: int):
    torch.manual_seed(seed)
    qweight = torch.randint(0, 2**31 - 1, (input_size, output_size // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    qzeros = torch.randint(0, 2**31 - 1, (input_size // GROUP_SIZE, output_size // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    scales = torch.rand((input_size // GROUP_SIZE, output_size), dtype=torch.float16, device=DEVICE)
    return qweight, qzeros, scales


def main():
    print("Compiling v3 kernel ...")
    v3_kernel = load(name="awq_dequantize_v3", sources=["awq_dequant_v3.cu"], verbose=False)

    qweight, qzeros, scales = make_weights(1536, 8960, seed=42)

    print("\n--- First call for this shape (expect autotune to run) ---")
    first_call_time = _timed_call(lambda: smart_awq_dequantize(qweight, scales, qzeros, v3_kernel))
    print(f"First call total time (including autotune): {first_call_time:.2f}ms")

    print("\n--- Second call, same shape (expect cache hit, fast) ---")
    second_call_time = _timed_call(lambda: smart_awq_dequantize(qweight, scales, qzeros, v3_kernel))
    print(f"Second call total time: {second_call_time:.4f}ms")

    print(f"\nCache now contains: {_block_size_cache}")
    print(f"Cache persisted to disk at: {os.path.abspath(CACHE_FILE)}")
    print(f"First call was {first_call_time / second_call_time:.0f}x slower than second call "
          f"-- this is the expected, accepted one-time autotune cost.")
    print("Run this script again -- the shape above should now load from disk")
    print("and skip autotuning entirely, even in a brand new process.")

    print("\n--- New, different shape (expect autotune to run again) ---")
    qweight2, qzeros2, scales2 = make_weights(8960, 1536, seed=99)
    third_call_time = _timed_call(lambda: smart_awq_dequantize(qweight2, scales2, qzeros2, v3_kernel))
    print(f"Third call (new shape) total time: {third_call_time:.2f}ms")
    print(f"Cache now contains: {_block_size_cache}")


if __name__ == "__main__":
    main()