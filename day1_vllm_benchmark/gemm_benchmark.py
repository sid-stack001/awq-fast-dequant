import time
import statistics

import torch
from vllm import _custom_ops as ops

HIDDEN_SIZE = 1536
INTERMEDIATE_SIZE = 8960
GROUP_SIZE = 128
WEIGHT_BITS = 4
PACK_FACTOR = 32 // WEIGHT_BITS  # = 8

NUM_LAYERS = 28
NUM_TOKENS = 16          
NUM_WARMUP_RUNS = 3
NUM_TIMED_RUNS = 8
DEVICE = "cuda"

# Measured reference point from Day 1: 1221.5ms / 128 generated tokens
DAY1_MEASURED_MS_PER_STEP = 1221.5 / 128


def make_weights(input_size: int, output_size: int):
    qweight = torch.randint(
        0, 2**31 - 1,
        (input_size, output_size // PACK_FACTOR),
        dtype=torch.int32, device=DEVICE,
    )
    qzeros = torch.randint(
        0, 2**31 - 1,
        (input_size // GROUP_SIZE, output_size // PACK_FACTOR),
        dtype=torch.int32, device=DEVICE,
    )
    scales = torch.rand(
        (input_size // GROUP_SIZE, output_size),
        dtype=torch.float16, device=DEVICE,
    )
    return qweight, qzeros, scales


def run_one_simulated_decode_step(attn_weights, mlp_up_weights, mlp_down_weights, x):
    aq_w, aq_z, aq_s = attn_weights
    up_w, up_z, up_s = mlp_up_weights
    down_w, down_z, down_s = mlp_down_weights

    for _ in range(NUM_LAYERS):
        for _ in range(4):
            ops.awq_gemm(x, aq_w, aq_s, aq_z, PACK_FACTOR)

        for _ in range(2):
            ops.awq_gemm(x, up_w, up_s, up_z, PACK_FACTOR)

        ops.awq_gemm(x, down_w, down_s, down_z, PACK_FACTOR)


def main():
    print("Building per-layer weight sets ...")
    attn_weights = make_weights(HIDDEN_SIZE, HIDDEN_SIZE)
    mlp_up_weights = make_weights(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    mlp_down_weights = make_weights(HIDDEN_SIZE, HIDDEN_SIZE) 

    x = torch.rand((NUM_TOKENS, HIDDEN_SIZE), dtype=torch.float16, device=DEVICE)

    total_calls = NUM_LAYERS * 7
    print(f"Simulating {NUM_LAYERS} layers x 7 GEMM calls = {total_calls} calls per step\n")

    # Warm-up: NOT timed, absorbs kernel compile/cache overhead
    for _ in range(NUM_WARMUP_RUNS):
        run_one_simulated_decode_step(attn_weights, mlp_up_weights, mlp_down_weights, x)
        torch.cuda.synchronize()

    # Timed runs: ONE synchronize at the end of each full step, not per call
    latencies_ms = []
    for _ in range(NUM_TIMED_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        run_one_simulated_decode_step(attn_weights, mlp_up_weights, mlp_down_weights, x)

        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000)

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms)
    per_call_ms = mean_ms / total_calls

    print(f"Simulated full-step GEMM time: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
    print(f"Implied per-call marginal cost (pipelined): {per_call_ms:.4f}ms")
    print(f"\nDay 1 measured full decode step time: {DAY1_MEASURED_MS_PER_STEP:.3f}ms")
    print(f"GEMM compute as % of measured total step time: "
          f"{100 * mean_ms / DAY1_MEASURED_MS_PER_STEP:.1f}%")


if __name__ == "__main__":
    main()