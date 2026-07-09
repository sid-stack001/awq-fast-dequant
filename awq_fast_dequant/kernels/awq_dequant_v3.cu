#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void awq_dequantize_v3_kernel(
    const int32_t* __restrict__ qweight,   // [K, M/8]
    const at::Half* __restrict__ scales,   // [K/group_size, M]
    const int32_t* __restrict__ zeros,     // [K/group_size, M/8]
    int K,
    int M,
    int group_size,
    at::Half* __restrict__ output          // [K, M]
) {
    int packed_cols = M / 8;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_packed_elements = K * packed_cols;
    if (idx >= total_packed_elements) return;

    int k = idx / packed_cols;
    int packed_col = idx % packed_cols;
    int group = k / group_size;

    int32_t packed_weight = qweight[idx];                          
    int32_t packed_zero = zeros[group * packed_cols + packed_col]; 

    const int reverse_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int shift = reverse_order[j] * 4;
        int raw_weight = (packed_weight >> shift) & 0xF;
        int raw_zero = (packed_zero >> shift) & 0xF;

        int m = packed_col * 8 + j;
        float scale = static_cast<float>(scales[group * M + m]);
        float dequantized = (static_cast<float>(raw_weight) - static_cast<float>(raw_zero)) * scale;

        output[k * M + m] = at::Half(dequantized);
    }
}

torch::Tensor awq_dequantize_v3_cuda(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int block_size
) {
    TORCH_CHECK(qweight.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
                "All tensors must be on CUDA");
    TORCH_CHECK(qweight.dtype() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(zeros.dtype() == torch::kInt32, "zeros must be int32");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");

    int K = qweight.size(0);
    int M = scales.size(1);
    int group_size = K / scales.size(0);

    auto output = torch::empty({K, M},
        torch::TensorOptions().dtype(torch::kFloat16).device(qweight.device()));

    int packed_cols = M / 8;
    int total_packed_elements = K * packed_cols;
    const int num_blocks = (total_packed_elements + block_size - 1) / block_size;

    awq_dequantize_v3_kernel<<<num_blocks, block_size>>>(
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
    m.def("awq_dequantize_v3", &awq_dequantize_v3_cuda, "Optimized AWQ dequantization (CUDA)");
}
