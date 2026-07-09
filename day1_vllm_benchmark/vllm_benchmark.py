import time
import json
import statistics
from dataclasses import dataclass, asdict

import torch
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
QUANTIZATION = "awq"         
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
NUM_WARMUP_RUNS = 3          
NUM_TIMED_RUNS = 5           
OUTPUT_TOKENS = 128          
PROMPT = "Explain the difference between throughput and latency in model serving."

RESULTS_PATH = f"benchmark_results_{QUANTIZATION}.json"


@dataclass
class BatchResult:
    batch_size: int
    mean_latency_ms: float
    std_latency_ms: float
    mean_throughput_tokens_per_sec: float
    std_throughput_tokens_per_sec: float
    raw_latencies_ms: list


def build_prompts(batch_size: int) -> list:
    return [PROMPT] * batch_size


def timed_generate(llm: LLM, prompts: list, sampling_params: SamplingParams) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()

    llm.generate(prompts, sampling_params)

    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000  # ms


def benchmark_batch_size(llm: LLM, batch_size: int) -> BatchResult:
    prompts = build_prompts(batch_size)
    sampling_params = SamplingParams(
        temperature=0.0,          
        max_tokens=OUTPUT_TOKENS,
        min_tokens=OUTPUT_TOKENS,  
    )


    for _ in range(NUM_WARMUP_RUNS):
        timed_generate(llm, prompts, sampling_params)

    # Timed runs
    latencies_ms = []
    for _ in range(NUM_TIMED_RUNS):
        latencies_ms.append(timed_generate(llm, prompts, sampling_params))

    total_tokens_per_run = batch_size * OUTPUT_TOKENS
    throughputs = [total_tokens_per_run / (lat_ms / 1000) for lat_ms in latencies_ms]

    return BatchResult(
        batch_size=batch_size,
        mean_latency_ms=statistics.mean(latencies_ms),
        std_latency_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        mean_throughput_tokens_per_sec=statistics.mean(throughputs),
        std_throughput_tokens_per_sec=statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        raw_latencies_ms=latencies_ms,
    )


def main():
    print(f"Loading {MODEL_NAME} ...")
    llm = LLM(
        model=MODEL_NAME,
        quantization=QUANTIZATION,
        max_model_len=512,
        gpu_memory_utilization=0.75,   
        dtype="float16",
    )

    results = []
    for batch_size in BATCH_SIZES:
        print(f"\nBenchmarking batch_size={batch_size} ...")
        try:
            result = benchmark_batch_size(llm, batch_size)
            results.append(result)
            print(
                f"  latency: {result.mean_latency_ms:.1f}ms ± {result.std_latency_ms:.1f}ms  |  "
                f"throughput: {result.mean_throughput_tokens_per_sec:.1f} ± "
                f"{result.std_throughput_tokens_per_sec:.1f} tok/s"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={batch_size} — stopping sweep here.")
            torch.cuda.empty_cache()
            break

    with open(RESULTS_PATH, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()