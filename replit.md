# CUDA C Project

## Overview
This project demonstrates basic CUDA C programming with a parallel vector addition example. The program showcases GPU-accelerated computing using NVIDIA's CUDA framework.

**Created:** November 15, 2025  
**Type:** CUDA C / GPU Computing

## Project Purpose
- Educational example of CUDA programming
- Demonstrates parallel computing on GPU
- Shows memory management between CPU and GPU
- Provides template for CUDA C projects

## Current State
Initial setup with a working vector addition example that:
- Adds two vectors of 1,000,000 elements in parallel on GPU
- Includes error checking and result verification
- Displays GPU device properties
- Demonstrates proper CUDA memory management

## Project Structure
```
.
├── vector_add.cu       # Main CUDA C source file with kernel
├── Makefile           # Build configuration for nvcc
├── README.md          # Detailed setup and usage instructions
├── .gitignore         # Git ignore rules for CUDA/C projects
└── replit.md          # This file
```

## Recent Changes
- **2025-11-15**: Initial project setup
  - Created vector addition CUDA kernel
  - Added Makefile for easy compilation
  - Created comprehensive documentation
  - Set up project structure

## Technical Details

### CUDA Kernel
The `vectorAdd` kernel performs parallel addition:
```c
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
```
- Runs on GPU with 256 threads per block
- Each thread processes one vector element
- Uses CUDA error checking macros

### Memory Pattern
1. Allocate on CPU (host)
2. Allocate on GPU (device) 
3. Copy data CPU → GPU
4. Execute kernel on GPU
5. Copy results GPU → CPU
6. Free all memory

## Requirements
- NVIDIA GPU with CUDA support (Compute Capability 5.0+)
- CUDA Toolkit (nvcc compiler)
- Compatible GPU drivers

## Important Notes
**⚠️ Replit Limitation**: This code is designed to run on systems with NVIDIA GPUs and the CUDA Toolkit installed. The standard Replit environment does not have CUDA-capable GPUs, so this code serves as a template and learning resource but cannot execute in this environment.

To run this code:
1. Use a local machine with NVIDIA GPU
2. Use cloud GPU services (Google Colab, AWS, Azure, etc.)
3. Use university/research computing clusters

## User Preferences
None specified yet.

## Architecture Decisions
- **Language**: CUDA C (.cu extension)
- **Vector Size**: 1,000,000 elements (configurable)
- **Thread Configuration**: 256 threads per block
- **Data Type**: float (32-bit floating point)
- **Error Handling**: CUDA_CHECK macro for comprehensive error checking
- **Architecture Target**: sm_50 (Maxwell and newer GPUs)

## Next Steps (Future Enhancements)
- Matrix multiplication example
- Performance timing and benchmarking
- CPU vs GPU performance comparison
- Shared memory optimization examples
- CUDA streams for concurrent execution
- More complex kernels (reduction, scan, etc.)
