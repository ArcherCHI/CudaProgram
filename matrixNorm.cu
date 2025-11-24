/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

/* Program Parameters */
#define N 6              /* Matrix size */
#define numThreads 16       /* Number of threads per block */

const size_t matrixSize = N * N * sizeof(float);

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

// Print A and B matrices
void printMatrices() {
    printf("Printing Sequential Matrices\n");
    printf("A = \n");
    int x, y;
    for ( x=0; x < N; x++ ){
        for ( y = 0; y < N; y++ ){
            printf("%f ", A[x][y] );
        } printf("\n");
    }
    printf("B = \n");
    for ( x=0; x < N; x++ ){
        for ( y = 0; y < N; y++ ){
            printf("%f ", B[x][y] );
        } printf("\n");
    }     printf("\n");
}

void printParallelMatrices( float* m1, float* m2 ){
    printf("Printing Parallel Matrices\n");
    printf("A = \n");
    int x, y;
    for ( x=0; x < N; x++ ){
        for ( y = 0; y < N; y++ ){
            printf("%f ", m1[x*N + y] );
        } printf("\n");
    }
    printf("B = \n");
    for ( x=0; x < N; x++ ){
        for ( y = 0; y < N; y++ ){
            printf("%f ", m2[x*N + y] );
        } printf("\n");
    }
}

void sequentialMatrixNorm() {
    int row, col;
    float mu, sigma; // Mean and Standard Deviation

    printf("Sequential Computing...\n");

    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B[row][col] = 0.0;
            else
                B[row][col] = (A[row][col] - mu) / sigma;
        }
    }

}

/* Kernel function */

__global__ void parallelMatrixNorm( float *devA, float *devB ) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    // float mu, sigma; // Mean and Standard Deviation
    float mu, sigma;

    if ( row < N && col < N ) {
        mu = 0.0;
        int r;
        for ( r = 0; r < N; r++ )
            mu += devA[r * N + col];
        mu /= (float) N;
        __syncthreads();
        
        sigma = 0.0;
        for ( r = 0; r < N; r++)
            sigma += powf( devA[ r * N + col ] - mu, 2.0 );
        sigma /= (float) N;
        __syncthreads();
        
        sigma = sqrt(sigma);
        for ( r = 0; r < N; r++ ) {
            int idx = r * N + col;
            if (sigma == 0.0)
                devB[idx] = 0.0;
            else {
                float currVal = devA[idx];
                devB[idx] = ( currVal - mu ) / sigma;
            }
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


    const int numBlocks = (N + numThreads - 1) / numThreads;
    
    // Define block and grid size
    dim3 dimBlock( numThreads, numThreads );
    dim3 dimGrid( numBlocks,  numBlocks );
    
    // 1. Allocate memory space in host (CPU) for data
    float *hostA, *hostB;    // host data
    hostA = (float*) malloc( matrixSize );
    hostB = (float*) malloc( matrixSize );
    if ( hostA == NULL || hostB == NULL ) 
        printf("Error allocating memory on host\n");
    
    int x, y;
    for ( x=0; x < N; x++ ){
        for ( y = 0; y < N; y++ ){
            hostA[x*N + y] = A[x][y];
            hostB[x*N + y] = 0.0;
        }
    }
    
    // 2. Allocate memory space in device (GPU) for data
    float *deviceA, *deviceB;    // device data
    cudaMalloc((void**) &deviceA, matrixSize );
    cudaMalloc((void**) &deviceB, matrixSize );
    
    // 3. Copy data from host to device
    cudaMemcpy( deviceA, hostA, matrixSize, cudaMemcpyHostToDevice );
    cudaMemcpy( deviceB, hostB, matrixSize, cudaMemcpyHostToDevice );
    
    // Print initial matrices ( for debugging )
    printMatrices();
    printParallelMatrices( deviceA, deviceB );
    
    /* Matrix Normalization */
    // 4. Execute kernel function in device   
    printf("Parallel Computing...\n");
    parallelMatrixNorm<<<dimGrid, dimBlock>>>(deviceA, deviceB);
    sequentialMatrixNorm();
    
    // 5. Copy data from device to host
    cudaMemcpy( hostA, deviceA, matrixSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( hostB, deviceB, matrixSize, cudaMemcpyDeviceToHost );

    printParallelMatrices( hostA, hostB );
    printMatrices();

    // 6. Free memory space in device
    cudaFree(deviceA);
    cudaFree(deviceB);
    
    // 7. Free memory space in host
    free(hostA);
    free(hostB);

    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);


    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");

    // 8. Return
    exit(0);
}
