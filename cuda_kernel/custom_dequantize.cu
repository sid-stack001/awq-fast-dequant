// Custom CUDA implementation of AWQ dequantization.
//
// Formula 
// awq_triton.py reference implementation):
//
//   For weight at input-row k, output-column m:
//     group        = k // group_size
//     packed_word  = qweight[k, m // 8]         (one int32, 8 packed 4-bit values)
//     j            = m % 8                       (which of the 8 packed values)
//     reverse_order = [0, 4, 1, 5, 2, 6, 3, 7]    (AWQ's interleaved storage order)
//     shift        = reverse_order[j] * 4
//     raw_weight   = (packed_word >> shift) & 0xF
//     raw_zero     = (zeros[group, m // 8] >> shift) & 0xF   (same unpacking)
//     scale        = scales[group, m]                          (NOT packed)
//     output       = (raw_weight - raw_zero) * scale

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void awq_dequantize_kernel(
    const int32_t* __restrict__ qweight,   // [K, M/8]
    const at::Half* __restrict__ scales,   // [K/group_size, M]
    const int32_t* __restrict__ zeros,     // [K/group_size, M/8]
    int K,
    int M,
    int group_size,
    at::Half* __restrict__ output          // [K, M]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = K * M;
    if (idx >= total_elements) return;

    // Recover which (row, col) of the OUTPUT matrix this thread computes.
    int k = idx / M;
    int m = idx % M;

    int group = k / group_size;
    int packed_col = m / 8;   // which int32 in qweight holds our value
    int j = m % 8;            // which of the 8 packed slots within it

    // AWQ's interleaved unpacking order -- NOT sequential
    // Derived from vLLM's reference kernel.
    const int reverse_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    int shift = reverse_order[j] * 4;

    int32_t packed_weight = qweight[k * (M / 8) + packed_col];
    int32_t packed_zero = zeros[group * (M / 8) + packed_col];

    int raw_weight = (packed_weight >> shift) & 0xF;
    int raw_zero = (packed_zero >> shift) & 0xF;

    float scale = static_cast<float>(scales[group * M + m]);
    float dequantized = (static_cast<float>(raw_weight) - static_cast<float>(raw_zero)) * scale;

    output[idx] = at::Half(dequantized);
}

torch::Tensor awq_dequantize_cuda(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros
) {
    TORCH_CHECK(qweight.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
                "All tensors must be on CUDA");
    TORCH_CHECK(qweight.dtype() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(zeros.dtype() == torch::kInt32, "zeros must be int32");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");

    int K = qweight.size(0);
    int M = scales.size(1);
    int group_size = K / scales.size(0);

    TORCH_CHECK(qweight.size(1) == M / 8, "qweight shape mismatch with scales");
    TORCH_CHECK(zeros.size(0) == K / group_size && zeros.size(1) == M / 8,
                "zeros shape mismatch");

    auto output = torch::empty({K, M},
        torch::TensorOptions().dtype(torch::kFloat16).device(qweight.device()));

    int total_elements = K * M;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    awq_dequantize_kernel<<<num_blocks, threads_per_block>>>(
        qweight.data_ptr<int32_t>(),
        reinterpret_cast<at::Half*>(scales.data_ptr<at::Half>()),
        zeros.data_ptr<int32_t>(),
        K, M, group_size,
        output.data_ptr<at::Half>()
    );

    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("awq_dequantize", &awq_dequantize_cuda, "Custom AWQ dequantization (CUDA)");
}
