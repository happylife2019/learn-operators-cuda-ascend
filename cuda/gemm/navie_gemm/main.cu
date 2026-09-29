#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

constexpr int M = 1024
constexpr int N =  1024
constexpr int K =  1024
#define TILE_SIZE 16

__global__ void matmulNaive(float *A,float *B, float*C, intM,int N,int K){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row<M && col<N){
        float sum = 0.0f;
        for(int k=0;k<K;k++){
            sum+=A[row*K+k]*B[k*N+col];
        }
        C[row*N+col] = sum;
    }
}

void initMatrix(float *mat ,int rows,int cols){
    for(int i=0;i<rows*cols;i++){
        mat[i] = (float)(rand()%100)/10.0f;
    }
}

float computeGFLOPS(float time_ms,int M, int N, int K){
    double flops = 2.0*M*N*K;
    return float(flops/(time_ms*1e-3)/1e-9);
}

bool verifyResult(float *C_cpu,float* C_gpu,int rows,int cols){
    for(int i = 0;i<rows*cols;i++){
        if(fabs(C_gpu[i]-C_cpu[i])>1e-6){
            printf("mismatch at index %d",i);
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
    float* h_C_ref = (float*)malloc(sizeA);
    initMatrix(h_A,M,K);
    initMatrix(h_B,K,N);
    for(int i = 0;i<M;i++){
        for(int j = 0;j<N;j++){
            float sum = 0.0f;
            for(int k = 0;k<K;k++){
                sum+= h_A[i*K+k]*h_B[k*N+j];
            }
            h_C_ref[i*N+j] = sum;
        }
    }
    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,sizeA);
    cudaMalloc(&d_B,sizeB);
    cudaMalloc(&d_C,sizeC);
    cudaMemcpy(d_A,h_A,sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,sizeA,cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(16,16);

    dim3 grid((N + 15) / 16, (M + 15) / 16);

    cudaMemset(d_C, 0, sizeC);
    cudaEventRecord(start);

    matmulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive_ms = 0.0f;
    cudaEventElapsedTime(&time_naive_ms, start, stop);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\n========== 性能对比 ==========\n");
    printf("[Naive] 耗时: %.3f ms, GFLOPS: %.2f\n",
        time_naive_ms, computeGFLOPS(time_naive_ms, M, N, K));
    printf("正确性: %s\n",
        verifyResult(h_C_ref, h_C, M, N) ? "PASS" : "FAIL");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}