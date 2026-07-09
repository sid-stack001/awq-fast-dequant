# Troubleshooting Log: vLLM on WSL2 (RTX 3060 Laptop, 6GB VRAM)

Environment issues encountered while getting `vllm==0.6.3.post1` serving
`Qwen/Qwen2.5-1.5B-Instruct-AWQ` running under WSL2 on a memory-constrained
consumer GPU. Each issue was a genuine dependency, ABI, or runtime
mismatch rather than a code defect; the fixes generalize to similar
WSL2 and CUDA and vLLM setups.

## Environment (final, working)

- OS: WSL2 / Ubuntu (Windows host)
- GPU: NVIDIA GeForce RTX 3060 Laptop, 6GB VRAM
- Driver: CUDA UMD version 13.3 (backward-compatible with cu12.x toolkits)
- Python: 3.11 (conda env)
- torch: 2.4.0+cu121
- vllm: 0.6.3.post1
- transformers: 4.46.3

## Issue 1 — VRAM OOM at engine startup

**Symptom:**
```
ValueError: Free memory on device cuda:0 (5.0/6.0 GiB) on startup is less
than desired GPU memory utilization (0.85, 5.1 GiB).
```

**Root cause:** requested `gpu_memory_utilization=0.85` assumed the full 6GB
was free. In practice, WSL2's GPU driver overhead and background processes
left only ~5.0GiB free before vLLM even started.

**Fix:** lowered `gpu_memory_utilization` to 0.65 (~3.9GiB), safely inside
actual free memory with margin. Checked `nvidia-smi` first to confirm what
was resident on the GPU before assuming the budget.

**Lesson:** never assume `gpu_memory_utilization` is a fraction of *total*
VRAM in practice — it's a fraction of what's free at init time, and that
number is smaller than the spec sheet says, especially under WSL2.

## Issue 2 — Missing C++ compiler (Triton/torch.compile)

**Symptom:** `Failed to find C compiler` — vLLM's V1 engine uses
`torch.compile` + Triton to generate fused kernels at runtime, which
requires a system C++ toolchain.

**Fix:** `sudo apt install build-essential`

## Issue 3 — Missing nvcc (FlashInfer JIT)

**Symptom:** `RuntimeError: Could not find nvcc` — FlashInfer's sampling
kernels attempt on-the-fly CUDA JIT compilation.

**Attempted fix:** installed `nvidia-cuda-toolkit` via apt — this pulled a
toolkit version with 100+ template compilation errors due to strict version
mismatches against the installed PyTorch/vLLM wheels.

**Actual fix:** deferred FlashInfer entirely. It's an optional accelerated
attention backend, not a hard vLLM requirement — the default attention
backend works without it. Chasing exact nvcc/toolkit/wheel alignment for an
optional dependency wasn't worth the cost at this stage.

## Issue 4 — Dependency resolver drift (uv/pip silently swapping versions)

**Symptom:** installing `flashinfer-python` wheels from a custom index
silently forced a `torch` downgrade to 2.9.1, which broke `vllm==0.24.0`'s
strict requirement on `torch==2.11.0`.

**Root cause:** PyPI does not allow a package to pin a custom index for its
own transitive dependencies. When a sub-dependency declares a `torch==X`
requirement that doesn't match what's installed, pip/uv can silently
resolve to a different index and swap out a deliberately-chosen build.

**Fix (short-term):** forced targeted reinstall with explicit pins and
`--index-strategy unsafe-best-match`.

**Real fix (see "Full Reset" below):** stopped fighting the bleeding-edge
`vllm==0.24.0` V1 engine's fast-moving dependency chain and pinned to an
older, more stable release line instead.

## Issue 5 — `libcudart.so.13` ImportError

**Symptom:** `vllm==0.24.0`'s PyPI wheel was built expecting CUDA 13
runtime libraries (`libcudart.so.13`), but the environment (aligned to
`cu126` torch wheels) only had `libcudart.so.12`.

**Rejected fix:** symlinking `libcudart.so.13 → libcudart.so.12`. This
"works" only if vLLM never calls a runtime symbol that changed between CUDA
major versions — a real risk of silent wrong results or a crash deep into a
long run, not just a startup failure. Not used in the final setup.

**Actual fix:** abandoned this dependency combination entirely (see below).

## Full Reset — pinning a coherent, tested version set

Rather than keep patching a fast-moving `vllm==0.24.0` (V1 engine) install
against WSL2, did a clean environment reset with everything pinned to a
known-compatible release line:

```bash
conda deactivate
conda env remove -n vllm
conda create -n vllm python=3.11 -y
conda activate vllm

# Install torch FIRST, verify before touching vLLM
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# -> 2.4.0+cu121 12.1 True

# Older, more stable vLLM release line — matches torch 2.4.0 / cu121 by default
pip install vllm==0.6.3.post1
```

**Key discipline:** verified `pip show torch vllm | grep Version` after
*every* install step, not just at the end — this is what catches silent
resolver drift before it costs another multi-hour debugging session.

**Confirmed via vLLM's own docs:** the `0.6.x` release line was built by
default against CUDA 12.1 and public PyTorch release versions — so pinning
`torch==2.4.0+cu121` alongside `vllm==0.6.3.post1` isn't a guess, it's the
documented pairing for that release.

## Issue 6 — `transformers` API break (`all_special_tokens_extended`)

**Symptom:**
```
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended.
Did you mean: 'num_special_tokens_to_add'?
```

**Root cause:** `transformers>=5.0` removed `all_special_tokens_extended`,
but `vllm==0.6.3.post1`'s tokenizer wrapper (from late 2024) still calls it
internally. pip's resolver installed a `transformers` version newer than
what that vLLM release was built against — same class of problem as Issue 4,
different library.

**Fix:** pinned `transformers==4.46.3` — contemporaneous with
`vllm==0.6.3.post1`'s release (Oct/Nov 2024), predating the breaking change.

```bash
pip install "transformers==4.46.3"
```

## Results — vLLM AWQ throughput/latency sweep, RTX 3060 (6GB)

Full sweep, batch sizes 1→32, no OOM. `gpu_memory_utilization=0.65`,
`quantization=awq`, `max_model_len=512`, fixed 128-token generation,
5 timed runs per batch size after 3 discarded warm-up runs.

| Batch | Latency (mean ± std) | Throughput (mean ± std) | Throughput scaling vs prior |
|---|---|---|---|
| 1  | 821.9ms ± 8.1ms   | 155.8 ± 1.5 tok/s   | — |
| 2  | 848.2ms ± 4.6ms   | 301.8 ± 1.7 tok/s   | 1.94x |
| 4  | 852.1ms ± 4.5ms   | 600.9 ± 3.2 tok/s   | 1.99x |
| 8  | 960.2ms ± 6.0ms   | 1066.5 ± 6.6 tok/s  | 1.77x |
| 16 | 1221.5ms ± 13.5ms | 1676.7 ± 18.5 tok/s | 1.57x |
| 32 | 1876.6ms ± 34.3ms | 2183.3 ± 39.2 tok/s | 1.30x |

GPU block accounting from vLLM's own log: 4163 GPU KV-cache blocks available
at this config, reporting max concurrency of 130x for 512-token requests —
confirms 32 concurrent sequences never approached the KV-cache ceiling, so
the falloff above is compute-bound, not memory-capacity-bound.

**Reading the scaling curve:** batch 1→4 is near-linear — latency barely
moves while throughput scales ~2x per doubling, meaning the GPU is
underutilized at low batch sizes and absorbing extra concurrent requests
almost for free. Scaling degrades steadily from batch 8 onward (1.77x →
1.57x → 1.30x per doubling), marking the transition from
memory-bandwidth-bound to compute-bound execution on a 6GB Ampere card.
That inflection point (roughly batch 8) is a natural target for the Day 2
C++/CUDA optimization work — it's where a hand-tuned kernel would have the
most headroom to matter.

**On the outlier runs visible in the raw logs:** one run out of every 8 at
batch=2 and batch=32 was markedly slower (e.g. 6.29 tok/s vs. a typical
~28 tok/s at batch=2) — visible in the console output but *not* reflected
in the reported mean/std. This is because the outliers landed in the
3 discarded warm-up runs, not the 5 timed runs. Confirms the warm-up
discard is doing its job: absorbing one-off scheduling jitter (likely a
WSL2 host-OS interruption or CUDA graph re-capture) rather than letting it
corrupt the reported statistics.

**Next cheap experiment, flagged directly by vLLM's own log output:** the
engine explicitly warned `awq quantization is not fully optimized yet` and
recommended `quantization=awq_marlin` for faster inference on this model.
Re-running the same sweep with `awq_marlin` is a low-effort way to quantify
the real cost of the suboptimal quantization kernel — worth doing before
moving to the C++/CUDA extension, since it's a one-line config change with
a directly comparable before/after number.

## AWQ vs AWQ_Marlin — kernel comparison

Same sweep, same model, same hardware, only `quantization` changed.

| Batch | AWQ throughput | AWQ_Marlin throughput | Improvement |
|---|---|---|---|
| 1  | 155.8 tok/s  | 159.6 tok/s  | +2.4%  |
| 2  | 301.8 tok/s  | 317.3 tok/s  | +5.1%  |
| 4  | 600.9 tok/s  | 619.4 tok/s  | +3.1%  |
| 8  | 1066.5 tok/s | 1172.5 tok/s | +9.9%  |
| 16 | 1676.7 tok/s | 2088.8 tok/s | +24.6% |
| 32 | 2183.3 tok/s | 2944.6 tok/s | +34.9% |

**The advantage is not flat — it grows with batch size.** At batch 1-4,
decode is memory-bandwidth-bound (dominated by moving weights from VRAM),
so kernel choice barely matters (+2-5%). From batch 8 onward, execution
shifts compute-bound and the gap widens sharply, reaching +34.9% at
batch 32 — the signature of a genuinely better GEMM kernel rather than a
constant-factor speedup.

**This also revises the earlier compute-bound-crossover finding.**
Per-doubling throughput scaling efficiency:

| Doubling | AWQ  | AWQ_Marlin |
|---|---|---|
| 8→16  | 1.57x | 1.78x |
| 16→32 | 1.30x | 1.41x |

`awq_marlin` doesn't just add a flat speedup — it delays the compute-bound
crossover to a higher batch size, staying closer to linear scaling further
out than the unoptimized `awq` kernel. The original conclusion ("bottleneck
emerges around batch 8") was specific to the `awq` kernel; with
`awq_marlin`, the same GPU sustains near-linear scaling further into the
batch sweep. Having both configurations measured, rather than just one,
is what makes this a defensible kernel-level finding rather than a single
unexplained number.

## Issue 7 — WSL2 disconnects and JIT compile hangs indefinitely

**Symptom:** VS Code's WSL remote connection failed
(`Couldn't install vscode server on remote server`), and separately, a
`torch.utils.cpp_extension.load()` call hung with no visible progress for
well past its expected 1-3 minute compile time, with no `nvcc` process
running (`ps aux | grep nvcc` returned nothing).

**Investigation:** `df -h` showed the Windows host `C:` drive at 99% full
(6.2GB free out of 496GB), while the WSL Linux filesystem itself had
934GB free. `free -h` showed memory was not under pressure. WSL2's own
virtualization layer and driver files live on the host drive, so a
near-full host drive can degrade or hang WSL2 operations even when the
Linux-side filesystem looks completely healthy.

**Fix applied:** freed space on the host `C:` drive, then `wsl --shutdown`
followed by a clean reconnect, then cleared `~/.cache/torch_extensions`
to remove any build artifacts from the interrupted compile before retrying.

**Honest note on causation:** after the fix, compilation succeeded. A
second, unverified explanation was also proposed (an agent's shell not
inheriting `conda activate`, causing `ninja` to be missing from `PATH`).
The two explanations were not tested independently — the disk cleanup and
a shell/environment change happened close together, so which one actually
resolved the hang is not fully isolated. A missing-`ninja`-on-`PATH`
failure would be expected to fail immediately with a clear error, not
hang silently with no `nvcc` process running, which is why the host-disk
explanation is considered more likely — but this is not confirmed with a
controlled, single-variable test.

**Lesson:** when a WSL2 environment misbehaves in ways that don't match
the Linux-side symptoms (plenty of disk, plenty of memory, no obvious
error), check the Windows host's own disk space before assuming the
problem is inside the Linux environment.

## General lessons

1. **Pin the whole stack together, not package by package.** Installing
   `torch`, then `vllm`, then extras one at a time and letting the resolver
   pick "latest compatible" at each step is how you end up with a torch
   build expecting CUDA 13 next to a library expecting CUDA 12.
2. **Bleeding-edge releases (vLLM's V1 engine, latest `transformers`) drag
   in the least-tested dependency combinations.** An older, more
   battle-tested release line trades some features for a much higher chance
   of "it just works" on non-standard hardware (WSL2, 6GB consumer GPU).
3. **Verify after every install, not just at the end.** `pip show <pkg> |
   grep Version` after each step catches silent version swaps immediately,
   when they're cheap to fix, instead of after the next failure downstream.
4. **WSL2-specific quirks are real and separate from CUDA/driver issues:**
   NVML incompatibility with `fork()` (forces `spawn` multiprocessing),
   reduced default memory allocation (needed a `.wslconfig` change), and
   background driver overhead eating into "free" VRAM reported by
   `nvidia-smi`.