import torch
from torch.utils.cpp_extension import load
from vllm import _custom_ops as ops

GROUP_SIZE = 128
PACK_FACTOR = 8
DEVICE = "cuda"

def main():
    v2_kernel = load(name="awq_dequantize_v2", sources=["awq_dequantize_v2.cu"], verbose=False)
    v3_kernel = load(name="awq_dequantize_v3", sources=["awq_dequant_v3.cu"], verbose=False)


    input_size = 1536
    output_size = 8960

    torch.manual_seed(42)
    qweight = torch.randint(0, 2**31 - 1, (input_size, output_size // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    qzeros = torch.randint(0, 2**31 - 1, (input_size // GROUP_SIZE, output_size // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    scales = torch.rand((input_size // GROUP_SIZE, output_size), dtype=torch.float16, device=DEVICE)

    torch.cuda.synchronize()

    ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
    
    v2_kernel.awq_dequantize_v2(qweight, scales, qzeros)
    
    v3_kernel.awq_dequantize_v3(qweight, scales, qzeros, 128)

    torch.cuda.synchronize()

if __name__ == "__main__":
    main()