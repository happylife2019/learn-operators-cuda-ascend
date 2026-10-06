#include "wmma_gemm.h"
#include <mma.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include<limits>

namespace{
namespace wmma = nvcuda::wmma 
constexpr int BM=128;
constexpr int BN=128;
constexpr int BK=32;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS/32;
constexpr int AS = BK+8;
constexpr int BS = BN+8;
constexpr uint32_t PLAN_COOKIE = 0x574d4d41u;
struct  OperandStorage{
    __half a[2][BM][AS];
    __half b[2][BK][BS];
};

union __aligned__(32) SharedStorage {
    OperandStorage operands;
    float output[WARPS][16*16];
}

static_assert(sizeof(SharedStorage) == 37888, "shared-memory budget changed");
static_assert(AS % 8 == 0 && BS % 8 == 0, "WMMA half stride alignment");

__device__ __forceinline__ void copy_async_16(__half* dst, const __half* src){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0],[1%], 16 ;"::"r"(address),"l"(src):"memory");
#else 
    *reinterpret_cast<uint4*>(dst) = reinterpret_cast<const uint4*>(src)
#endif
}

__device__ __forceinline__ void commit_copies(){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
    asm volatile("cp.async.commit_group;":::"memory");
#endif
}

__device__ __forceinline__ void wait_copies(){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
    asm volatile("cp.async.wait_group 0;":::"memory");
#endif
}


}