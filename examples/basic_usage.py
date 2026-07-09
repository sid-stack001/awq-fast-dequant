"""
Minimal usage example.

Run after installing the package (`pip install -e .` from the repo root).
"""

import torch
import awq_fast_dequant as awq

GROUP_SIZE = 128
PACK_FACTOR = 8


def make_example_weights(input_size: int, output_size: int):
    qweight = torch.randint(
        0, 2**31 - 1, (input_size, output_size // PACK_FACTOR),
        dtype=torch.int32, device="cuda",
    )
    qzeros = torch.randint(
        0, 2**31 - 1, (input_size // GROUP_SIZE, output_size // PACK_FACTOR),
        dtype=torch.int32, device="cuda",
    )
    scales = torch.rand(
        (input_size // GROUP_SIZE, output_size),
        dtype=torch.float16, device="cuda",
    )
    return qweight, qzeros, scales


if __name__ == "__main__":
    qweight, qzeros, scales = make_example_weights(1536, 8960)

    # Fixed block size (128), no autotuning:
    result_v3 = awq.dequantize_v3(qweight, scales, qzeros, block_size=128)
    print(f"v3 (fixed block size) output shape: {result_v3.shape}")

    # Automatic block size selection, cached per shape after the first call:
    result_smart = awq.smart_dequantize(qweight, scales, qzeros)
    print(f"smart_dequantize output shape: {result_smart.shape}")

    assert torch.allclose(result_v3, result_smart, atol=1e-2, rtol=1e-2)
    print("v3 and smart_dequantize outputs match, as expected -- same "
          "underlying kernel, different block size selection strategy.")