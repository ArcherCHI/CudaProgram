/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

/* Program Parameters */
#define N 6000  /* Matrix size */

/* Matrices */
volatile float A[N][N], B[N][N];


/* Initialize A and B*/
void initialize_inputs() {
    int row, col;

    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }

}


/* Kernel function */

__global__ void matrixNorm( float *devA, float *devB ) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    // float mu, sigma; // Mean and Standard Deviation
    __shared__ float mu, sigma;
    printf("Computing Serially.\n");

    if ( col < N && row < N ) {
        int idx = row * N + col;
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += devA[idx];
        mu /= (float) N;
        __syncthreads();
        
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf( devA[idx] - mu, 2.0);
        sigma /= (float) N;
        __syncthreads();
        
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            int idx2 = row * N + col;
            if (sigma == 0.0)
                devB[idx2] = 0.0;
            else
                devB[idx2] = ( devA[idx] - mu ) / sigma;
        }
    }

}

/* Basic CUDA program structure 
    1. Allocate memory space in host (CPU) for data
    2. Allocate memory space in device (GPU) for data
    3. Copy data from host to device
    4. Execute kernel function in device
        - with CUDA syntax that defines number of threads and their physical structure
    5. Copy data from device to host
    6. Free memory space in device
    7. Free memory space in host
    8. Return
*/



int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval start, stop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime;

    /* Initialize A and B */
    initialize_inputs();


    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start, &tzdummy);

    // Define block and grid size
    // const int numBlocks = 16;
    const int numThreads = 16;    // Threads per block
    dim3 dimBlock( numThreads, numThreads );
    // dim3 dimGrid( numBlocks, numBlocks );
    dim3 dimGrid( N / numThreads, N / numThreads );
    
    // 1. Allocate memory space in host (CPU) for data
    float *host;    // host data
    host = (float*)malloc(N*N*sizeof(float));
    
    // 2. Allocate memory space in device (GPU) for data
    float *deviceA, *deviceNorm;    // device data
    cudaMalloc((void**) &deviceA, N*N*sizeof(float));
    cudaMalloc((void**) &deviceNorm, N*N*sizeof(float));
    
    // 3. Copy data from host to device
    cudaMemcpy( deviceA, host, N*N*sizeof(float), cudaMemcpyHostToDevice );

    /* Matrix Normalization */
    // 4. Execute kernel function in device    
        // - with CUDA syntax that defines number of threads and their physical structure
    matrixNorm<<<dimGrid, dimBlock>>>(deviceA, deviceNorm);
    
    // 5. Copy data from device to host
    cudaMemcpy( host, deviceNorm, N*N*sizeof(float), cudaMemcpyDeviceToHost );

    // 6. Free memory space in device
    cudaFree(deviceA);
    cudaFree(deviceNorm);

    // 7. Free memory space in host
    free(host);
    
    // matrixNorm();


    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);


    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");

    exit(0);
}