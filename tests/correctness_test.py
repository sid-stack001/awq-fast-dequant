"""
Correctness tests for the CUDA kernels, checked against a pure-Python
reference implementation of the AWQ dequantization formula -- independent
of both the CUDA kernels themselves and of vLLM, so this doesn't just
check "does our kernel match vLLM's kernel" but "does our kernel match
the formula, derived and verified independently."

These tests require a CUDA-capable GPU and are skipped automatically when
one isn't available (e.g. on standard GitHub Actions runners, which don't
provide GPUs). See tests/test_autotune_cache.py for CPU-only tests that
run on every CI job regardless of hardware.
"""

import torch
import pytest

import awq_fast_dequant as awq

CUDA_AVAILABLE = torch.cuda.is_available()
skip_without_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="requires a CUDA-capable GPU"
)

# Deliberately small shapes: this is a correctness check, not a benchmark,
# and a small shape keeps the nested-loop reference implementation fast
# while still exercising multiple groups and multiple packed words.
GROUP_SIZE = 32
PACK_FACTOR = 8
TEST_SHAPES = [
    (32, 64),    # 1 group, 8 packed words per row
    (64, 128),   # 2 groups, 16 packed words per row
]


def reference_dequantize(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """
    Deliberately simple, unoptimized, one-value-at-a-time implementation
    of the AWQ dequantization formula, mirroring the hand-derived formula
    exactly: reverse_order lookup, shift, mask, subtract zero, multiply
    by scale. Intended to be obviously correct by inspection, not fast --
    this is a test utility, not a kernel.
    """
    K, packed_cols = qweight.shape
    M = scales.shape[1]
    group_size = K // scales.shape[0]
    reverse_order = [0, 4, 1, 5, 2, 6, 3, 7]

    output = torch.empty((K, M), dtype=torch.float16, device=qweight.device)

    for k in range(K):
        group = k // group_size
        for m in range(M):
            packed_col = m // 8
            j = m % 8
            shift = reverse_order[j] * 4

            raw_weight = (int(qweight[k, packed_col]) >> shift) & 0xF
            raw_zero = (int(zeros[group, packed_col]) >> shift) & 0xF
            scale = float(scales[group, m])

            output[k, m] = (raw_weight - raw_zero) * scale

    return output


def make_test_weights(input_size: int, output_size: int, seed: int, device: str):
    torch.manual_seed(seed)
    qweight = torch.randint(
        0, 2**31 - 1, (input_size, output_size // PACK_FACTOR),
        dtype=torch.int32, device=device,
    )
    qzeros = torch.randint(
        0, 2**31 - 1, (input_size // GROUP_SIZE, output_size // PACK_FACTOR),
        dtype=torch.int32, device=device,
    )
    scales = torch.rand(
        (input_size // GROUP_SIZE, output_size),
        dtype=torch.float16, device=device,
    )
    return qweight, qzeros, scales


@skip_without_cuda
@pytest.mark.parametrize("input_size,output_size", TEST_SHAPES)
def test_dequantize_v2_matches_reference(input_size, output_size):
    qweight, qzeros, scales = make_test_weights(input_size, output_size, seed=0, device="cuda")
    expected = reference_dequantize(qweight, scales, qzeros)
    actual = awq.dequantize_v2(qweight, scales, qzeros)
    assert torch.allclose(expected, actual, atol=1e-2, rtol=1e-2)


@skip_without_cuda
@pytest.mark.parametrize("input_size,output_size", TEST_SHAPES)
@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_dequantize_v3_matches_reference(input_size, output_size, block_size):
    qweight, qzeros, scales = make_test_weights(input_size, output_size, seed=1, device="cuda")
    expected = reference_dequantize(qweight, scales, qzeros)
    actual = awq.dequantize_v3(qweight, scales, qzeros, block_size=block_size)
    assert torch.allclose(expected, actual, atol=1e-2, rtol=1e-2)


@skip_without_cuda
def test_dequantize_v3_block_size_does_not_affect_correctness():
    """
    Block size controls how work is partitioned across GPU threads, not
    what is computed -- results should be identical regardless of which
    block size is used. This directly tests that assumption rather than
    just asserting it in a comment.
    """
    qweight, qzeros, scales = make_test_weights(64, 128, seed=2, device="cuda")
    result_64 = awq.dequantize_v3(qweight, scales, qzeros, block_size=64)
    result_256 = awq.dequantize_v3(qweight, scales, qzeros, block_size=256)
    assert torch.allclose(result_64, result_256, atol=1e-2, rtol=1e-2)


@skip_without_cuda
def test_smart_dequantize_matches_reference():
    qweight, qzeros, scales = make_test_weights(32, 64, seed=3, device="cuda")
    expected = reference_dequantize(qweight, scales, qzeros)
    actual = awq.smart_dequantize(qweight, scales, qzeros)
    assert torch.allclose(expected, actual, atol=1e-2, rtol=1e-2)
    awq.clear_cache()  # don't leave test-generated entries in the real cache file