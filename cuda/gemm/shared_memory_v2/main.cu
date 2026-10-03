#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

template <int BM, int BK>
__device__ __forceinline__
void load_tile_A(const float* A, float As[][BK],int by, int bk, int tid, int M, int K) {
    for (int t = tid; t < BM * BK; t += blockDim.x) {
        int i = t / BK;  // As 中的行
        int p = t % BK;  // As 中的列
        int r = by * BM + i;
        int c = bk + p;

        As[i][p] = (r < M && c < K) ? A[r * K + c] : 0.0f;
    }
}

void checkCuda(cudaError_t error, const char* step) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(error));
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

bool verifyResult(const float* C_cpu, const float* C_gpu, int rows, int cols) {
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

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling(const float* A, const float* B, float* C,
                                    int M, int N, int K);

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    const size_t sizeA = (size_t)M * K * sizeof(float);
    const size_t sizeB = (size_t)K * N * sizeof(float);
    const size_t sizeC = (size_t)M * N * sizeof(float);
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Host memory allocation failed\n");
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_ref);
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
    checkCuda(cudaMemset(d_C, 0, sizeC), "clear C");

    dim3 block((BM / TM) * (BN / TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start), "record start");
    sgemm_thread_tiling<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    checkCuda(cudaGetLastError(), "launch sgemm_thread_tiling");
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "wait for kernel");
    float time_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "measure time");

    checkCuda(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "copy C back");
    bool pass = verifyResult(h_C_ref, h_C, M, N);
    printf("[ThreadTiled] 耗时: %.3f ms, GFLOPS: %.2f\n",
           time_ms, computeGFLOPS(time_ms, M, N, K));
    printf("正确性: %s\n", pass ? "PASS" : "FAIL");

    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    checkCuda(cudaFree(d_A), "free A");
    checkCuda(cudaFree(d_B), "free B");
    checkCuda(cudaFree(d_C), "free C");
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    return pass ? 0 : 1;
}

template <int BK, int BN>
__device__ __forceinline__
void load_tile_B(const float* B, float Bs[][BN],int bx, int bk, int tid, int K, int N) {
    for (int t = tid; t < BK * BN; t += blockDim.x) {
        int p = t / BN;  // Bs 中的行
        int j = t % BN;  // Bs 中的列
        int r = bk + p;
        int c = bx * BN + j;
        Bs[p][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
    }
}

template <int BM,int BN, int BK,int TM,int TN>
__global__ void sgemm_thread_tiling(const float* A,const float* B ,float* C,int M,int N,int K){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int by= blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int thread_row = (tid/(BN/TN))*TM;
    int thread_col = (tid%(BN/TN))*TN;
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};
    for(int bk = 0;bk<K;bk+=BK){
        load_tile_A<BM, BK>(A, As, by, bk, tid, M, K);
        load_tile_B<BK, BN>(B, Bs, bx, bk, tid, K, N);
        __syncthreads();
        for(int k=0;k<BK;k++){
            for(int i=0;i<TM;i++){
                a_frag[i] = As[thread_row+i][k];
            }
            for(int j = 0;j<TN;j++){
                b_frag[j] = Bs[k][thread_col+j];
            }
            for(int i = 0;i<TM;i++){
                for(int j = 0;j<TN;j++){
                    c_frag[i][j] += a_frag[i]*b_frag[j];
                }
            }
        }
        __syncthreads();
    }
    for(int i = 0;i<TM;i++){
        for(int j = 0;j<TN;j++){
            if(by*BM+thread_row+i<M&&bx*BN+thread_col+j<N){
                C[(by*BM+thread_row+i)*N+bx*BN+thread_col+j] = c_frag[i][j];
            }
        }
    }
}
