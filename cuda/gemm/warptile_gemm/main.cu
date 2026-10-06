#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void checkCuda(cudaError_t error, const char* step) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

bool verifyResult(const float* ref, const float* actual, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        const float diff = fabsf(actual[i] - ref[i]);
        const float tolerance = 1e-3f + 1e-5f * fabsf(ref[i]);
        if (!(diff <= tolerance)) {
            printf("mismatch at index %d: CPU=%.8f GPU=%.8f diff=%.8f\n",
                   i, ref[i], actual[i], diff);
            return false;
        }
    }
    return true;
}

template <int BM, int BK>
__device__ __forceinline__
void load_tile_A(const float* A, float As[][BK], int by, int bk,
                 int tid, int M, int K) {
    for (int t = tid; t < BM * BK; t += blockDim.x) {
        const int i = t / BK;
        const int p = t % BK;
        const int row = by * BM + i;
        const int col = bk + p;
        As[i][p] = (row < M && col < K) ? A[row * K + col] : 0.0f;
    }
}

template <int BK, int BN>
__device__ __forceinline__
void load_tile_B(const float* B, float Bs[][BN], int bx, int bk,
                 int tid, int K, int N) {
    for (int t = tid; t < BK * BN; t += blockDim.x) {
        const int p = t / BN;
        const int j = t % BN;
        const int row = bk + p;
        const int col = bx * BN + j;
        Bs[p][j] = (row < K && col < N) ? B[row * N + col] : 0.0f;
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_warp_tiling(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    // 本例固定 256 线程：8 个 warp，每个 warp 为 4×8 个线程微块。
    // 8 个 warp 在 block 内排成 4×2；每个线程计算 TM×TN 个 C。
    static_assert(BM == 128 && BN == 128 && BK == 8 && TM == 8 && TN == 8,
                  "This teaching example uses 128x128x8 tiles and 8x8 outputs/thread");

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    // v2 是 tid/16、tid%16：一个 warp 的 32 个线程逻辑上排成 2×16。
    // 这里改成 4×8；线程总数、block tile 和每线程输出数都不变。
    const int warp_id = tid >> 5;       // tid / 32，范围 0..7
    const int lane_id = tid & 31;       // tid % 32，范围 0..31
    const int warp_row = warp_id >> 1;  // 4×2 warp 排布中的行
    const int warp_col = warp_id & 1;   // 4×2 warp 排布中的列
    const int lane_row = lane_id >> 3;  // 4×8 lane 排布中的行
    const int lane_col = lane_id & 7;   // 4×8 lane 排布中的列

    // 当前线程负责的 8×8 C 子块在 128×128 block tile 内的左上角。
    // 例如 tid=8：warp_id=0, lane_row=1, lane_col=0 => row_c=8, col_c=0。
    const int row_c = (warp_row * 4 + lane_row) * TM;
    const int col_c = (warp_col * 8 + lane_col) * TN;

    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        // Global -> Shared：与 v2 相同，每个线程协作搬运 A、B 的一部分。
        load_tile_A<BM, BK>(A, As, by, bk, tid, M, K);
        load_tile_B<BK, BN>(B, Bs, bx, bk, tid, K, N);
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            // Shared -> Register：按新的 row_c/col_c 取本线程需要的值。
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[row_c + i][k];
            }
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[k][col_c + j];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();  // 所有线程算完，才能覆写 As/Bs 装下一块。
    }

    for (int i = 0; i < TM; ++i) {
        const int row = by * BM + row_c + i;
        for (int j = 0; j < TN; ++j) {
            const int col = bx * BN + col_c + j;
            if (row < M && col < N) {
                C[row * N + col] = c_frag[i][j];
            }
        }
    }
}

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
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        return 1;
    }

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
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

    const dim3 block(256);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start), "record start");
    sgemm_warp_tiling<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    checkCuda(cudaGetLastError(), "launch sgemm_warp_tiling");
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "wait for kernel");
    float time_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "measure time");

    checkCuda(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "copy C back");
    const bool pass = verifyResult(h_C_ref, h_C, M, N);
    const double gflops = (2.0 * M * N * K) / (time_ms * 1e-3) / 1e9;
    printf("[WarpTiled] 耗时: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);
    printf("正确性: %s\n", pass ? "PASS" : "FAIL");

    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    checkCuda(cudaFree(d_A), "free A");
    checkCuda(cudaFree(d_B), "free B");
    checkCuda(cudaFree(d_C), "free C");
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return pass ? 0 : 1;
}
