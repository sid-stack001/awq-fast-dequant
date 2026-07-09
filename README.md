# Custom AWQ Dequantization Kernel

A CUDA kernel for AWQ 4-bit weight dequantization, built to investigate and
improve on a measured bottleneck in vLLM's inference pipeline. Developed and
benchmarked on a memory-constrained consumer GPU (RTX 3060 Laptop, 6GB
VRAM) under WSL2.

**Result:** a custom kernel, derived from vLLM's own reference
implementation and verified bit-for-bit against its output, achieves a
1.15x-1.5x speedup over vLLM's production AWQ dequantization kernel across
the three primary layer shapes in Qwen2.5-1.5B-Instruct. The result was
stress-tested across 15 independent trials per shape, randomized call
ordering, and confirmed with Nsight Compute hardware counters, not wall
clock timing alone.

This is a standalone benchmarking project, not a vLLM contribution. See
[Limitations](#limitations) for scope.

## Installation

Two separate setups, for two different purposes.

**To use the kernel as a library** (`awq_fast_dequant/`):

```bash
pip install -e .
python examples/basic_usage.py
```

Requires `nvcc` (CUDA toolkit, 12.0+ confirmed working) and `ninja`
available on `PATH`. The first call to `dequantize_v2`, `dequantize_v3`,
or `smart_dequantize` triggers a one-time JIT compilation (roughly
1-3 minutes); subsequent calls reuse the cached build.

**To reproduce the exact benchmark results** in this README and
`TROUBLESHOOTING.md` (`day1_vllm_benchmark/`, `cuda_kernel/`):

```bash
conda create -n vllm python=3.11 -y
conda activate vllm
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

See `requirements.txt` for exact pinned versions and
`TROUBLESHOOTING.md` for why these specific versions matter.

## Motivation

Built as applied preparation for ML infrastructure work: closing the gap
between using vLLM/PyTorch and understanding what runs on the GPU, and
producing a technical artifact that can be defended in detail, not just
described.

## Repository structure

```
day1_vllm_benchmark/
    vllm_benchmark.py              vLLM batch-size throughput/latency sweep
    gemm_benchmark.py              Isolated benchmark of vLLM's AWQ GEMM call
    profiling.py                   Engine-level profiling (see Dead Ends)
    benchmark_results.json         Raw sweep output, quantization=awq
    benchmark_results_awq_marlin.json   Raw sweep output, quantization=awq_marlin

cuda_kernel/
    cuda_test.cu / cuda_test.py    Minimal build toolchain verification
    custom_dequantize.cu           v1: naive kernel, one thread per output element
    awq_dequantize_v2.cu           v2: one thread per packed word (fixes redundant reads)
    awq_dequant_v3.cu              v3: block size exposed as a runtime parameter
    block_sweep.py                 Multi-shape, multi-trial stress test with block size sweep
    smart_dequantize.py            Autotuning wrapper with persistent cache
    ncu_profile.py                 Nsight Compute hardware profiling
    autotune_cache/                Persisted tuning decisions (generated, gitignored)

TROUBLESHOOTING.md                 Full environment debugging log
```

## Part 1: vLLM inference benchmarking

`vllm_benchmark.py` serves `Qwen/Qwen2.5-1.5B-Instruct-AWQ` and sweeps
batch sizes 1-32, using GPU-synchronized timing, discarded warm-up runs,
and mean/std over repeated measurements.

Getting a working environment took longer than writing the benchmark.
Six distinct issues required resolution: VRAM budgeting under WSL2, a
missing C++ toolchain, an nvcc/FlashInfer JIT failure, silent dependency
resolver drift, a CUDA major-version library mismatch, and a
`transformers` API break against a pinned vLLM release. Full root-cause
analysis in [TROUBLESHOOTING.md](TROUBLESHOOTING.md). The resolution in
each case followed one principle: pin the full dependency stack together
and verify after every install, rather than letting the package resolver
choose versions independently.

**Finding: compute-bound crossover.** Throughput scales near-linearly from
batch 1 to 4, then degrades from batch 8 onward (1.77x, 1.57x, 1.30x per
doubling). KV-cache accounting in vLLM's logs confirms this is a compute
limit rather than a memory-capacity limit; 32 concurrent sequences never
approached the cache ceiling.

**Finding: kernel choice matters more at scale.** Comparing `awq` against
`awq_marlin`: the gap is small at low batch size (2-5%, memory-bandwidth
bound) and widens sharply at high batch size (34.9% at batch 32,
compute-bound). `awq_marlin` also delays the compute-bound crossover to a
higher batch size. Since `awq`, not `awq_marlin`, is what actually ran
during the benchmark, this motivated Part 2.

## Part 2: Custom CUDA kernel

### Dead ends in profiling

Standard `torch.profiler` engine-level profiling was tried and abandoned
twice before switching to source-level analysis.

1. CUDA graphs hide the compute. vLLM pre-records execution into a
   replayed graph; graph-replayed kernels are invisible to PyTorch's
   op-level profiler. The profile attributed 86%+ of time to generic
   tensor-copy bookkeeping, an artifact of the measurement rather than
   the workload.
2. Disabling graphs (`enforce_eager=True`) did not resolve this. vLLM's
   AWQ compute runs as unnamed custom CUDA extensions, not named `aten::`
   operators, so the profiler showed only generic `cudaLaunchKernel`
   calls with no attribution.

The resolution was reading vLLM's source directly. `awq.py` identified the
function in use for this workload's token counts (`ops.awq_gemm`), and
`awq_triton.py`, a readable Triton reference implementation, provided the
exact dequantization formula, including AWQ's interleaved bit-packing
order (`[0,4,1,5,2,6,3,7]`, not sequential), which was hand-verified
against a worked example before any kernel code was written.

### Confirming the target

An isolated benchmark of `ops.awq_gemm` confirmed vLLM's 256-token
heuristic threshold for switching between GEMM strategies is well-tuned on
this GPU. A naive extrapolation from per-call isolated timing initially
exceeded the real measured decode step time; this was traced to artificial
serialization introduced by synchronizing before and after every isolated
call, which never happens in real pipelined execution. A corrected,
non-serialized benchmark found GEMM compute accounts for roughly 60% of
real decode step time, which justified the optimization work below.

### Kernel versions

- **v1 (naive):** one thread per output element. Verified bit-for-bit
  correct against vLLM's output (max absolute difference: 0.000000), but
  1.15x slower than vLLM's kernel.
- **v2 (fewer reads):** the inefficiency in v1 is that 8 threads
  independently re-read the same packed 32-bit word to extract one 4-bit
  value each. v2 uses one thread per packed word, reading it once and
  unpacking all 8 values in an unrolled loop. 1.87x faster than v1 in a
  controlled, single-variable comparison.
- **v3 (tunable block size):** exposes `threads_per_block` as a runtime
  parameter rather than a hardcoded constant, enabling the sweep and
  autotuning below.

### Stress testing

An early single-run comparison suggested a uniform 1.74x speedup over
vLLM. A more rigorous re-test, using 5+ independent random seeds, three
real layer shapes, and randomized call order on every timed run to rule
out positional bias, found the result did not hold uniformly. It was
strong and reproducible on the two MLP shapes and statistically
indistinguishable from vLLM on the smaller attention shape, where
kernel-launch overhead dominates over the memory-access optimization. A
subsequent run surfaced an outlier-contaminated mean on one shape (std
exceeding the mean); this was addressed by tracking median alongside mean
and flagging greater than 15% divergence between them.

Final result, 15 trials times 10 randomized-order runs per shape,
correctness-checked every trial:

| Shape | v2 vs vLLM | v3 (128 threads) vs vLLM |
|---|---|---|
| Attention (1536->1536) | 1.01x (not significant) | 1.15-1.17x |
| MLP up/gate (1536->8960) | 1.20-1.23x | 1.38-1.47x |
| MLP down (8960->1536) | 1.13-1.14x | 1.32-1.35x |

v3 at 64 threads per block was the fastest configuration in every shape
tested, up to 1.47x versus vLLM.

### Hardware validation

Wall-clock results were cross-checked against Nsight Compute performance
counters (`ncu --section SpeedOfLight`), which also corrected a bottleneck
misattribution in an earlier draft: vLLM's kernel is not DRAM-bandwidth
bound (31.51% DRAM utilization) but L1/TEX cache bound (96.26%
utilization, the actual saturated resource).

| Kernel | L1/TEX Throughput | DRAM Throughput |
|---|---|---|
| vLLM (baseline) | 96.26% | 31.51% |
| v2 | 88.96% | 46.36% |
| v3 (128 threads) | 84.57% | 56.58% |

The monotonic trend, cache pressure falling and DRAM utilization rising as
redundant reads are removed and block size shrinks, confirms the
mechanism rather than only the outcome.

### Autotuning

The best block size varies by shape; 64 threads wins most often but not
always, so a hardcoded constant does not generalize. `smart_dequantize.py`
implements shape-keyed autotuning: on first encountering a shape, it runs
a timing sweep across candidate block sizes and caches the result. Every
subsequent call for that shape is a cache lookup, measured at 53x faster
than the first call. The cache persists to
`autotune_cache/block_size_cache.json` and was confirmed to survive
process restarts.

## Limitations

- This is a standalone research repository, not a vLLM contribution. The
  kernel targets one packing format, is tested against synthetic random
  weights rather than trained model weights, and does not implement
  `split_k_iters`-style generality for very large matrices. Upstream
  contribution would require broader shape coverage, real-weight testing,
  and integration with vLLM's test suite.
- The autotuner's sweep is deliberately lighter weight than
  `block_sweep.py` (no randomized ordering, fewer runs). It is a fast
  production heuristic, not a publishable benchmark, and the two
  occasionally disagree by about 1%, within expected noise.
- The autotune cache is per-machine. Block size optima are likely
  hardware-specific and have not been validated on a second GPU.
- A three-tier edge/consumer/datacenter comparison, including a Jetson
  Nano, was considered and deferred. The original Jetson Nano's compute
  capability (5.3) cannot run vLLM or modern PyTorch, making this a
  separate project with its own toolchain (TensorRT/ONNX Runtime) rather
  than an extension of this one.
