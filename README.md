# AWQ Dequantization Kernel — A CUDA Learning Project

> **TL;DR**  This is a hands-on exploration of low-level GPU programming.
> The goal was to understand how a real production system (vLLM) works
> under the hood specifically how it unpacks compressed 4-bit weights on
> the GPU and to write a custom CUDA kernel doing the same thing from
> scratch. A speedup was observed under the specific test conditions used
> here, but **this is not a claim that the kernel is faster than vLLM in
> general** (see [Limitations](#limitations)).

Developed on a consumer RTX 3060 Laptop (6 GB VRAM) under WSL2. The
project covers reading vLLM source code, understanding GPU hardware
counters, writing and iterating on CUDA kernels, and building a rigorous
benchmark, step by step.

This is a standalone learning project, not a vLLM contribution or a
production-ready optimization.

---

## How it works

### Step 1 — Why do we need dequantization?

Modern LLMs are *quantized* to save memory: instead of storing each weight
as a 16-bit float (2 bytes), AWQ packs **eight** 4-bit values into a single
32-bit integer (4 bytes), cutting storage in half.

```
  Original weights (8 values, each 16-bit = 128 bits total):
  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
  │  w0    │  w1    │  w2    │  w3    │  w4    │  w5    │  w6    │  w7    │
  └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

  After AWQ packing (same 8 values crammed into 1 × 32-bit int = 32 bits):
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ w0 │ w4 │ w1 │ w5 │ w2 │ w6 │ w3 │ w7 │  ← interleaved order, NOT 0..7
  └────┴────┴────┴────┴────┴────┴────┴────┘
```

> AWQ stores values in an interleaved order `[w0, w4, w1, w5,
> w2, w6, w3, w7]`, not the obvious sequential order. This is a hardware
> alignment choice. Getting this wrong produces silently incorrect output.
> It was hand-verified against a worked example before writing any kernel code.

Before math can happen, the GPU must unpack each group of 8 values and
rescale them using a stored *scale* and *zero-point*. That is dequantization.
The formula for each weight:

```
output = (raw_weight - raw_zero) × scale
```

### Step 2 — Three kernel versions (each file is self-contained)

```
cuda_kernel/
  custom_dequantize.cu   ← v1: simplest possible, one thread per output float
  awq_dequantize_v2.cu   ← v2: fixes a redundant-read problem in v1
  awq_dequant_v3.cu      ← v3: makes block size a tunable parameter
```

All three produce **bit-for-bit identical output**. The only difference is
how many memory reads each one does.

**v1 — naive: one CUDA thread per output element**

Every thread is responsible for one output float. To get it, the thread
reads a 32-bit packed word and extracts its one 4-bit slot.

The inefficiency: eight threads independently re-read the *same* 32-bit
word to extract *different* slots. That is 8× the memory reads needed.

```
  Packed word (one int32 holding 8 values):
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ w0 │ w4 │ w1 │ w5 │ w2 │ w6 │ w3 │ w7 │
  └────┴────┴────┴────┴────┴────┴────┴────┘
    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
   T0   T1   T2   T3   T4   T5   T6   T7    ← 8 threads each reload the same word
```

**v2 — one thread per packed word (the fix)**

One thread reads the 32-bit word *once* and unpacks all 8 values in a
tight unrolled loop. Eight times fewer loads for `qweight` and `zeros`.

```
  Packed word:
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ w0 │ w4 │ w1 │ w5 │ w2 │ w6 │ w3 │ w7 │
  └────┴────┴────┴────┴────┴────┴────┴────┘
    ↑ one thread reads once, loops to extract all 8 values

  Result: 1.87× faster than v1 in a controlled single-variable comparison.
```

**v3 — tunable block size**

Same algorithm as v2, but `threads_per_block` is exposed as a runtime
parameter instead of a hardcoded constant. This enables the autotuning sweep
below.

### Step 3 — Autotuning: finding the best block size per shape

The fastest block size is not the same for every layer — 64 threads wins
most often but not always. `smart_dequantize.py` handles this automatically:

```
  First call with a new layer shape:
  ┌─────────────────────────────────────────────┐
  │  Sweep candidate block sizes: 64, 128, 256… │  ← one-time cost
  │  Pick fastest. Save result to JSON cache.   │
  └──────────────────┬──────────────────────────┘
                     │
  Every later call with the same shape:
  ┌──────────────────▼──────────────────────────┐
  │  Cache hit → launch directly with best size │  ← 53× faster than first call
  └─────────────────────────────────────────────┘

  Cache persists across process restarts: autotune_cache/block_size_cache.json
```

---

## Benchmark results (under specific test conditions)

> **Read this before interpreting the numbers.** All measurements were
> taken with **generated synthetic weights**, not real trained model
> weights. The layer shapes are based on Qwen2.5-1.5B-Instruct but may not
> represent the full diversity of real-world LLM workloads. vLLM's kernel
> is a mature, production-grade implementation optimized across a much
> wider range of hardware and use cases — **it is likely faster in general**.
> The numbers below describe what happened in *these* tests; they do not
> generalize.

Final benchmark: 15 trials × 10 randomized-order timed runs per shape,
correctness-checked every trial against vLLM's reference output.

| Layer shape | v2 vs vLLM | v3 (best block) vs vLLM |
|---|---|---|
| Attention (1536→1536) | 1.01× (no difference) | 1.15–1.17× |
| MLP up/gate (1536→8960) | 1.20–1.23× | 1.38–1.47× |
| MLP down (8960→1536) | 1.13–1.14× | 1.32–1.35× |

The speedup on these shapes, with these synthetic weights, is real and
reproducible — and the hardware counters explain *why* (cache pressure
falls, DRAM utilization rises as redundant reads are removed). But that
is not the same as saying this kernel would win in a real deployment.

**Hardware counter cross-check (Nsight Compute):**

| Kernel | L1/TEX cache pressure | DRAM utilization |
|---|---|---|
| vLLM baseline | 96.26% ← saturated | 31.51% |
| v2 | 88.96% | 46.36% |
| v3 (128 threads) | 84.57% | 56.58% |

---

## Installation

Two separate setups for two different purposes.

**To use the kernel as a library** (`awq_fast_dequant/`):

```bash
pip install -e .
python examples/basic_usage.py
```

Requires `nvcc` (CUDA toolkit 12.0+) and `ninja` on your `PATH`. The first
call triggers a one-time JIT compile (~1–3 minutes); subsequent calls reuse
the cached build.

**To reproduce the exact benchmark results** (`day1_vllm_benchmark/`,
`cuda_kernel/`):

```bash
conda create -n vllm python=3.11 -y
conda activate vllm
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

See `requirements.txt` for exact pinned versions and `TROUBLESHOOTING.md`
for why these specific versions matter.

---

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

awq_fast_dequant/                  Installable Python package wrapping the kernels
examples/basic_usage.py            Minimal runnable example

TROUBLESHOOTING.md                 Full environment debugging log
```

---

## Motivation

Built as applied preparation for ML infrastructure work: closing the gap
between using vLLM/PyTorch and understanding what runs on the GPU, and
producing a technical artifact that can be defended in detail, not just
described.

---

## Part 1: vLLM inference benchmarking

`vllm_benchmark.py` serves `Qwen/Qwen2.5-1.5B-Instruct-AWQ` and sweeps
batch sizes 1–32, using GPU-synchronized timing, discarded warm-up runs,
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
limit rather than a memory-capacity limit.

**Finding: kernel choice matters more at scale.** Comparing `awq` against
`awq_marlin`: the gap is small at low batch size (2–5%, memory-bandwidth
bound) and widens sharply at high batch size (34.9% at batch 32,
compute-bound). Since `awq` is what ran during the benchmark, this
motivated Part 2.

---

## Part 2: Custom CUDA kernel

### Dead ends in profiling

Standard `torch.profiler` engine-level profiling was tried and abandoned
twice before switching to source-level analysis.

1. **CUDA graphs hide the compute.** vLLM pre-records execution into a
   replayed graph; graph-replayed kernels are invisible to PyTorch's
   op-level profiler. The profile attributed 86%+ of time to generic
   tensor-copy bookkeeping — an artifact of the measurement, not the workload.
2. **Disabling graphs didn't help.** vLLM's AWQ compute runs as unnamed
   custom CUDA extensions, not named `aten::` operators, so the profiler
   showed only generic `cudaLaunchKernel` calls with no attribution.

The resolution was reading vLLM's source directly. `awq.py` identified the
function in use (`ops.awq_gemm`), and `awq_triton.py` provided the exact
dequantization formula — including AWQ's interleaved bit-packing order
(`[0,4,1,5,2,6,3,7]`), which was hand-verified against a worked example
before any kernel code was written.

### Confirming the target

An isolated benchmark of `ops.awq_gemm` confirmed vLLM's 256-token
heuristic threshold for switching between GEMM strategies is well-tuned on
this GPU. A corrected, non-serialized benchmark found GEMM compute accounts
for roughly 60% of real decode step time, which justified the optimization work.

---

## Limitations (important)

- **The speedup numbers do not generalize.** The benchmark used randomly
  generated synthetic weights, a single GPU (RTX 3060 Laptop), and three
  specific layer shapes from one small model. Real trained weights,
  different quantization distributions, larger matrices, or different
  hardware could easily flip the result. vLLM's kernel is battle-tested
  across many more conditions than this project covers.
- **Synthetic weights are not real weights.** AWQ's optimization relies on
  the statistical distribution of real trained weights. A kernel that
  performs well on `torch.randint` inputs may behave differently on actual
  quantized model weights.
- **This is a learning project, not a production optimization.** The point
  was to understand how the pieces fit together — quantization formats,
  CUDA thread hierarchies, GPU cache behavior, profiling tools — not to
  ship a faster kernel. Writing something that is correct, measurable, and
  explainable was the goal.
- **Autotuner is a fast heuristic.** The autotune sweep is lighter than
  `block_sweep.py` (no randomized ordering, fewer runs). It occasionally
  disagrees with the full benchmark by ~1%, within expected noise.
- **Cache is per-machine.** Block size optima are hardware-specific and
  have not been validated on a second GPU.

---

## Possible next steps

1. Validate against real trained AWQ weights; current checks confirm the
   unpacking formula is exact but have not stressed edge cases (all-zero
   weights, extreme values).
2. Package with an ahead-of-time CUDA build rather than JIT-compiled scripts.
3. Test whether autotuned block sizes transfer across other Ampere-family GPUs.