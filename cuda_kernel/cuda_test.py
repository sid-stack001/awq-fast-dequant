import torch
from torch.utils.cpp_extension import load

print("Compiling CUDA extension (this calls nvcc -- may take a minute) ...")
minimal_cuda = load(
    name="cuda_test",
    sources=["cuda_test.cu"],
    verbose=True,
)

print("\nCompilation succeeded. Testing correctness ...")

x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
print(f"Input:    {x}")

result = minimal_cuda.add_one(x)
print(f"Output:   {result}")

expected = x + 1.0
print(f"Expected: {expected}")

assert torch.allclose(result, expected), "Mismatch! Kernel did not compute correctly."
print("\n✓ CUDA toolchain confirmed working: nvcc compiled, pybind11 wrapped, "
      "kernel executed, and result matches expected output.")