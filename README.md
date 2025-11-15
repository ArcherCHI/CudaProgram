# CUDA C Vector Addition Example

A simple CUDA C program demonstrating GPU-accelerated parallel computing with vector addition.

## Overview

This project contains a basic CUDA C implementation that:
- Allocates memory on both CPU (host) and GPU (device)
- Transfers data between host and device
- Executes parallel computation on the GPU
- Verifies the results and displays GPU properties

## Files

- `vector_add.cu` - Main CUDA C source file with vector addition kernel
- `Makefile` - Build configuration for nvcc compiler
- `README.md` - This file

## Requirements

To compile and run this program, you need:

1. **NVIDIA GPU** with CUDA support (Compute Capability 5.0 or higher)
2. **CUDA Toolkit** (version 8.0 or later)
   - Includes the `nvcc` compiler
   - Download from: https://developer.nvidia.com/cuda-downloads
3. **Compatible GPU drivers**
4. **GCC or compatible C compiler** (usually installed with CUDA Toolkit)

## How to Build

```bash
# Compile the program
make

# Or compile directly with nvcc
nvcc -O2 -arch=sm_50 -o vector_add vector_add.cu
```

## How to Run

```bash
# Run the compiled program
./vector_add

# Or use make
make run
```

## Expected Output

```
Initializing vectors with 1000000 elements...
Copying data to GPU...
Launching CUDA kernel with 3907 blocks and 256 threads per block...
Copying result back to CPU...
Verifying results...
SUCCESS! All 1000000 elements computed correctly.
Sample results:
  a[0] + b[0] = 0.0 + 0.0 = 0.0
  a[100] + b[100] = 100.0 + 200.0 = 300.0
  a[999999] + b[999999] = 999999.0 + 1999998.0 = 2999997.0

CUDA device properties:
  Device name: [Your GPU Name]
  Compute capability: X.X
  Total global memory: X.XX GB
  Max threads per block: 1024
```

## Understanding the Code

### CUDA Kernel
```c
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
```
- `__global__` indicates this function runs on the GPU
- Each thread computes one element: `c[idx] = a[idx] + b[idx]`
- Thread index is calculated using block and thread IDs

### Memory Management
1. **Host allocation**: `malloc()` for CPU memory
2. **Device allocation**: `cudaMalloc()` for GPU memory
3. **Data transfer**: `cudaMemcpy()` between host and device
4. **Cleanup**: `free()` and `cudaFree()`

### Kernel Launch
```c
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
```
- Triple angle brackets `<<<>>>` indicate kernel launch
- 256 threads per block
- Enough blocks to cover all elements

## Troubleshooting

### "nvcc: command not found"
- Install CUDA Toolkit
- Add CUDA bin directory to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

### "no CUDA-capable device detected"
- Ensure you have an NVIDIA GPU
- Update GPU drivers
- Check GPU compatibility: https://developer.nvidia.com/cuda-gpus

### Compilation errors
- Check CUDA Toolkit version compatibility
- Adjust `-arch=sm_XX` flag for your GPU architecture
- Run `nvcc --version` to verify installation

## Clean Up

```bash
# Remove compiled files
make clean
```

## Next Steps

To learn more about CUDA programming:
- Try modifying the vector size
- Implement matrix multiplication
- Experiment with different block/thread configurations
- Add timing to compare GPU vs CPU performance
- Explore shared memory for optimization

## Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Developer Zone](https://developer.nvidia.com/cuda-zone)
