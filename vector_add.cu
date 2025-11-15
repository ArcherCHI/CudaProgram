#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
// Executed on GPU, each thread computes one element
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main(void) {
    int n = 1000000;  // Vector size
    size_t bytes = n * sizeof(float);
    
    // Host (CPU) memory allocation
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize vectors on host
    printf("Initializing vectors with %d elements...\n", n);
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }
    
    // Device (GPU) memory allocation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));
    
    // Copy data from host to device
    printf("Copying data to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel on GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching CUDA kernel with %d blocks and %d threads per block...\n", 
           blocksPerGrid, threadsPerBlock);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    printf("Copying result back to CPU...\n");
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify result
    printf("Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            if (errors < 10) {  // Print first 10 errors
                printf("Error at index %d: expected %f, got %f\n", 
                       i, expected, h_c[i]);
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("SUCCESS! All %d elements computed correctly.\n", n);
        printf("Sample results:\n");
        printf("  a[0] + b[0] = %.1f + %.1f = %.1f\n", h_a[0], h_b[0], h_c[0]);
        printf("  a[100] + b[100] = %.1f + %.1f = %.1f\n", h_a[100], h_b[100], h_c[100]);
        printf("  a[%d] + b[%d] = %.1f + %.1f = %.1f\n", 
               n-1, n-1, h_a[n-1], h_b[n-1], h_c[n-1]);
    } else {
        printf("FAILED! Found %d errors.\n", errors);
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("\nCUDA device properties:\n");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    
    return 0;
}
