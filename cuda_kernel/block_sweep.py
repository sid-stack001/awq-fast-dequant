import random
import time
import statistics

import torch
from torch.utils.cpp_extension import load
from vllm import _custom_ops as ops

GROUP_SIZE = 128
PACK_FACTOR = 8
DEVICE = "cuda"

NUM_TRIALS = 15           
NUM_TIMED_RUNS_PER_TRIAL = 10
NUM_WARMUP_RUNS = 3

SHAPES = [
    (1536, 1536, "attention-shaped (q/k/v/o, approx)"),
    (1536, 8960, "MLP up/gate-shaped"),
    (8960, 1536, "MLP down-shaped"),
]


def make_weights(input_size: int, output_size: int, seed: int):
    torch.manual_seed(seed)
    qweight = torch.randint(0, 2**31 - 1, (input_size, output_size // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    qzeros = torch.randint(0, 2**31 - 1, (input_size // GROUP_SIZE, output_size // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    scales = torch.rand((input_size // GROUP_SIZE, output_size), dtype=torch.float16, device=DEVICE)
    return qweight, qzeros, scales


def timed_call(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000


def run_randomized_order_benchmark(fns: dict, num_runs: int):
    results = {name: [] for name in fns}
    names = list(fns.keys())

    for _ in range(num_runs):
        order = names[:]
        random.shuffle(order)
        for name in order:
            results[name].append(timed_call(fns[name]))

    return results


def summarize(latencies: list) -> tuple:
    return statistics.mean(latencies), statistics.median(latencies), statistics.stdev(latencies)


def main():
    print("Compiling kernels ...")
    v1_kernel = load(name="awq_dequantize_custom", sources=["custom_dequantize.cu"], verbose=False)
    v2_kernel = load(name="awq_dequantize_v2", sources=["awq_dequantize_v2.cu"], verbose=False)
    v3_kernel = load(name="awq_dequantize_v3", sources=["awq_dequant_v3.cu"], verbose=False)

    all_shape_results = {}

    for input_size, output_size, description in SHAPES:
        print(f"\n{'='*80}")
        print(f"Shape: {input_size} -> {output_size}  ({description})")
        print(f"{'='*80}")

        keys = ["vllm", "v1", "v2", "v3_64", "v3_128", "v3_256"]
        aggregated_data = {k: [] for k in keys}
        any_correctness_failure = False

        for trial in range(NUM_TRIALS):
            seed = trial * 1000 + input_size
            qweight, qzeros, scales = make_weights(input_size, output_size, seed)

            # --- Correctness check: every kernel, every trial ---
            reference_output = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            v1_output = v1_kernel.awq_dequantize(qweight, scales, qzeros)
            v2_output = v2_kernel.awq_dequantize_v2(qweight, scales, qzeros)
            v3_64_out = v3_kernel.awq_dequantize_v3(qweight, scales, qzeros, 64)
            v3_128_out = v3_kernel.awq_dequantize_v3(qweight, scales, qzeros, 128)
            v3_256_out = v3_kernel.awq_dequantize_v3(qweight, scales, qzeros, 256)

            v1_ok = torch.allclose(reference_output, v1_output, atol=1e-2, rtol=1e-2)
            v2_ok = torch.allclose(reference_output, v2_output, atol=1e-2, rtol=1e-2)
            v3_64_ok = torch.allclose(reference_output, v3_64_out, atol=1e-2, rtol=1e-2)
            v3_128_ok = torch.allclose(reference_output, v3_128_out, atol=1e-2, rtol=1e-2)
            v3_256_ok = torch.allclose(reference_output, v3_256_out, atol=1e-2, rtol=1e-2)

            if not (v1_ok and v2_ok and v3_64_ok and v3_128_ok and v3_256_ok):
                print(f"  Trial {trial}: CORRECTNESS FAILURE "
                      f"(v1={v1_ok}, v2={v2_ok}, v3_64={v3_64_ok}, "
                      f"v3_128={v3_128_ok}, v3_256={v3_256_ok})")
                any_correctness_failure = True
                continue

            fns = {
                "vllm": lambda: ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0),
                "v1":   lambda: v1_kernel.awq_dequantize(qweight, scales, qzeros),
                "v2":   lambda: v2_kernel.awq_dequantize_v2(qweight, scales, qzeros),
                "v3_64":  lambda: v3_kernel.awq_dequantize_v3(qweight, scales, qzeros, 64),
                "v3_128": lambda: v3_kernel.awq_dequantize_v3(qweight, scales, qzeros, 128),
                "v3_256": lambda: v3_kernel.awq_dequantize_v3(qweight, scales, qzeros, 256),
            }

            # --- Warm-up: every kernel ---
            for _ in range(NUM_WARMUP_RUNS):
                for fn in fns.values():
                    fn()

            # --- Randomized-order timed runs ---
            trial_results = run_randomized_order_benchmark(fns, NUM_TIMED_RUNS_PER_TRIAL)

            for k in keys:
                aggregated_data[k].extend(trial_results[k])

            print(f"  Trial {trial}: vllm={statistics.mean(trial_results['vllm']):.4f}ms | "
                  f"v2={statistics.mean(trial_results['v2']):.4f}ms | "
                  f"v3_128={statistics.mean(trial_results['v3_128']):.4f}ms")

        if any_correctness_failure:
            print("  Skipping summary for this shape due to correctness failure(s).")
            continue

        # --- Aggregate: mean, median, std clearly labeled as separate stats ---
        print(f"\n  Aggregate over {NUM_TRIALS} trials x {NUM_TIMED_RUNS_PER_TRIAL} runs:")
        shape_summary = {}
        for k in keys:
            mean_val, median_val, std_val = summarize(aggregated_data[k])
            shape_summary[k] = (mean_val, median_val, std_val)

        vllm_mean = shape_summary["vllm"][0]
        for k in keys:
            mean_val, median_val, std_val = shape_summary[k]
            print(f"    {k:7s}: mean={mean_val:.4f}ms  median={median_val:.4f}ms  "
                  f"std={std_val:.4f}ms  ({vllm_mean/mean_val:.2f}x vs vllm mean)")

        for k in keys:
            mean_val, median_val, std_val = shape_summary[k]
            if mean_val > 0 and abs(mean_val - median_val) / mean_val > 0.15:
                print(f"    WARNING: {k} mean and median differ by >15% "
                      f"({mean_val:.4f}ms vs {median_val:.4f}ms) -- likely outlier-skewed, "
                      f"trust median over mean for this one.")

        all_shape_results[description] = shape_summary

    print(f"\n{'='*80}")
    print("FINAL SUMMARY ACROSS ALL MODEL LAYER SHAPES")
    print(f"{'='*80}")
    for desc, results in all_shape_results.items():
        vllm_mean, vllm_median, _ = results["vllm"]
        v2_mean, v2_median, _ = results["v2"]
        v3_128_mean, v3_128_median, _ = results["v3_128"]
        print(f"  {desc}:")
        print(f"    -> v2   : {vllm_mean/v2_mean:.2f}x (mean) / "
              f"{vllm_median/v2_median:.2f}x (median) relative to vLLM")
        print(f"    -> v3_128: {vllm_mean/v3_128_mean:.2f}x (mean) / "
              f"{vllm_median/v3_128_median:.2f}x (median) relative to vLLM")


if __name__ == "__main__":
    main()