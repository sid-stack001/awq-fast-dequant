import torch
from torch.profiler import profile, ProfilerActivity
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
QUANTIZATION = "awq"         
BATCH_SIZE = 16                
OUTPUT_TOKENS = 128
PROMPT = "Explain the difference between throughput and latency in model serving."


def main():
    print(f"Loading {MODEL_NAME} ...")
    llm = LLM(
        model=MODEL_NAME,
        quantization=QUANTIZATION,
        max_model_len=512,
        gpu_memory_utilization=0.65,
        dtype="float16",
        enforce_eager=True,   # disable CUDA graph capture so the profiler can
                               # see individual ops (attention, dequant, matmul)
                               # instead of one opaque replayed graph.
    )

    prompts = [PROMPT] * BATCH_SIZE
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=OUTPUT_TOKENS,
        min_tokens=OUTPUT_TOKENS,
    )

    # One untimed warm-up call
    print("Warm-up run (not profiled) ...")
    llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()

    print("Profiled run ...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()

    # Sort by total CUDA time
    print("\nTop 20 operations by CUDA time:\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    prof.export_chrome_trace("profiler_trace.json")
    print("\nChrome trace written to profiler_trace.json (open at chrome://tracing)")


if __name__ == "__main__":
    main()