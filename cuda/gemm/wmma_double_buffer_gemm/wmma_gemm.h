#pragma once 
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
typedef struct WmmaGemmPlan{
    uint32_t cookie;
    int device;
    int max_grid_x;
} WmmaGemmPlan;

#ifdef __cplusplus
extern "C"{
#endif

cudaError_t wmma_gemm_init(WmmaGemmPlan* plan);
cudaError_t wmma_gemm_launch(const WmmaGemmPlan* plan,const __half* A, const __half* B, float* C,int M,int N,int K
                            int64_t lda,int64_t ldb, int64_t ldc,float alpha,float beta,cudaStream_t stream);



#ifdef __cplusplus
}
#endif