// Minimal CUDA kernel: adds 1 to every element of a float tensor.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void add_one_kernel(float* data, int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        data[idx] = data[idx] + 1.0f;
    }
}

torch::Tensor add_one_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    torch::Tensor output = input.clone();
    int num_elements = output.numel();
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;


    add_one_kernel<<<num_blocks, threads_per_block>>>(
        output.data_ptr<float>(), num_elements
    );
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one_cuda, "Add 1 to every element (CUDA)");
}