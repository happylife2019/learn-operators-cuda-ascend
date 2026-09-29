#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

constexpr int M = 1024;
constexpr int N =  1024;
constexpr int K =  1024;
#define TILE_SIZE 16

__global__ void matmulTiled(float* A,float* B,float* C,int M,int N,int K){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int rowStart = blockIdx.y*TILE_SIZE;
    int colStart = blockIdx.x*TILE_SIZE;
    float sum = 0.0f;
    for(int tile = 0;tile<(K+TILE_SIZE-1)/TILE_SIZE;tile++){
        int kStart = tile*TILE_SIZE;
        if(rowStart+ty<M && kStart+tx<K){
            As[ty][tx] = A[(rowStart+ty)*K+kStart+tx];
        }else{
            As[ty][tx]=0.0f;
        }
        if(kStart+ty<K && colStart+tx<N){
            Bs[ty][tx] = B[(kStart+ty)*N+colStart+tx];
        }else{
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        for(int k = 0;k<TILE_SIZE;k++){
            sum+= As[ty][k]*Bs[k][tx];
        }
        __syncthreads();
    }
    int row = rowStart+ty;
    int col = colStart+tx;
    if(row<M && col<N){
        C[row*N+col] = sum;
    }
}

void checkCuda(cudaError_t error, const char* step){
    if(error!=cudaSuccess){
        fprintf(stderr,"%s:%s\n",step,cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

float computeGFLOPS(float time_ms, int m, int n, int k) {
    double flops = 2.0 * m * n * k;
    return (float)(flops / (time_ms * 1e-3) / 1e9);
}

bool verifyResult(const float* C_cpu, const float* C_gpu,
                  int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(C_gpu[i] - C_cpu[i]);
        float tolerance = 1e-3f + 1e-5f * fabsf(C_cpu[i]);

        if (!(diff <= tolerance)) {
            printf("mismatch at index %d: CPU=%.8f GPU=%.8f diff=%.8f\n",
                   i, C_cpu[i], C_gpu[i], diff);
            return false;
        }
    }
    return true;
}

int main(){
    size_t sizeA = M*K*sizeof(float);
    size_t sizeB = K*N*sizeof(float);
    size_t sizeC = M*N*sizeof(float);
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void**)&d_A, sizeA), "cudaMalloc A");
    checkCuda(cudaMalloc((void**)&d_B, sizeB), "cudaMalloc B");
    checkCuda(cudaMalloc((void**)&d_C, sizeC), "cudaMalloc C");
    checkCuda(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice), "copy B");
    checkCuda(cudaMemset(d_C,0,sizeC),"clear C");
    dim3 block(TILE_SIZE,TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/(TILE_SIZE),(M+TILE_SIZE-1)/(TILE_SIZE));
    cudaEvent_t start,stop;
    checkCuda(cudaEventCreate(&start),"create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start), "record start");
    matmulTiled<<<grid,block>>>(d_A,d_B,d_C,M,N,K);
    checkCuda(cudaGetLastError(),"last matmulTiled");
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop),"wait for kernel");
    float time_tiled_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&time_tiled_ms,start,stop),"measure time");


    checkCuda(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost),"copy C back");
    bool pass = verifyResult(h_C_ref, h_C, M, N);
    printf("[Tiled] 耗时: %.3f ms, GFLOPS: %.2f\n",time_tiled_ms, computeGFLOPS(time_tiled_ms, M, N, K));
    printf("正确性: %s\n", pass ? "PASS" : "FAIL");
    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    checkCuda(cudaFree(d_A),"free A");
    checkCuda(cudaFree(d_B),"free B");
    checkCuda(cudaFree(d_C),"free C");
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    return pass?0:1;
}
