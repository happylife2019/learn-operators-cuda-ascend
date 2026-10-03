# v2：Thread-Tiled FP32 GEMM

## 相比 v1 改了什么

计算 `C[M×N] = A[M×K] × B[K×N]`。v1 用 `16×16` 个线程计算 `16×16` 个输出，每线程累加一个 `C` 元素。本版一个 block 计算 `128×128` 个输出：256 个线程各负责连续的 `8×8` 个输出，并用自己的 `c_frag[8][8]` 累加；它不是 Shared Memory 中的 C 数组。

每次沿 K 方向前进 8，线程协作把 `A` 的 `128×8` 和 `B` 的 `8×128` 子块搬入 Shared Memory。每一步 `k`，每个线程先从 Shared Memory 取 8 个 A 值和 8 个 B 值到 `a_frag`、`b_frag`，再对自己的 64 个 C 累加器做外积更新。完成所有 K 子块后，线程把结果写回 Global Memory。与 v1 同时改变了 tile 大小、每线程输出数和访存方式，现有数据不能单独归因于其中一项。

## 编译和运行

在 AutoDL 的仓库根目录执行，目标 GPU 为 RTX 3080 Ti (`sm_86`)：

```bash
nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas -v \
  cuda/gemm/shared_memory_v2/main.cu -o /root/autodl-tmp/thread_tiled_gemm
/root/autodl-tmp/thread_tiled_gemm
```

程序当前测试 FP32、`M=N=K=1024`。CPU 参考实现逐元素比较，容差为 `1e-3 + 1e-5 × |C_cpu|`。CUDA Event 包围一次 kernel launch；不计分配、H2D/D2H 拷贝或 CPU 参考计算。GFLOPS 按 `2×M×N×K / 时间` 计算，并非硬件计数器读数。

## 2026-10-03 实测

| 来源 | 结果 | 含义 |
| --- | --- | --- |
| 普通运行的 CUDA Event | `0.671 ms`，`3201.76 GFLOPS`，正确性 PASS | 单次 kernel 计时及由此计算的吞吐量 |
| `nvcc -Xptxas -v` | 94 registers/thread，8192 bytes Shared/block，0 spill loads/stores | 编译器报告的资源用量；不是运行时瓶颈分析 |
| Nsight Systems 运行中的 CUDA Event | `0.728 ms`，`2949.84 GFLOPS`，正确性 PASS | profiler 环境下程序自己的单次计时 |
| Nsight Systems `cuda_gpu_kern_sum` | 1 次 kernel，`593046 ns`（约 `0.593 ms`） | GPU 时间线中的该次 kernel duration；与 CUDA Event 的测量口径不同 |
| Nsight Systems `cuda_gpu_trace` | grid `8×8×1`，block `256×1×1`，94 registers/thread，静态 Shared `0.008 MB` | 确认了启动配置与资源用量 |

普通运行的 `0.671 ms` 与 Nsight Systems 的 `0.593 ms` **不是同一测量口径**，不能据此说 profiler 让 kernel 变快。旧 v1 曾测到约 `0.849 ms` 中位数；`0.849/0.671 ≈ 1.27×` 仅是跨次运行的初步对比，尚不能归因或视为稳定加速比。当前 Nsight Systems 报告没有提供 Shared Memory bank conflict、指令吞吐或 stall 原因等硬件计数器，不能据此解释具体瓶颈。

## Nsight Systems 复现

AutoDL 当前实例的 `nsys` 不在 `PATH` 中，实际路径如下：

```bash
mkdir -p /root/autodl-tmp/nsys_reports
/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx --sample=none --force-overwrite=true \
  -o /root/autodl-tmp/nsys_reports/thread_tiled \
  /root/autodl-tmp/thread_tiled_gemm
/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys stats \
  -r cuda_gpu_kern_sum -r cuda_gpu_trace \
  /root/autodl-tmp/nsys_reports/thread_tiled.nsys-rep
```

此 AutoDL 容器上的 Nsight Compute 报 `ERR_NVGPUCTRPERM`，暂时不能用它采集硬件计数器；不要把 Nsight Systems 的时间线数据当作这些计数器。报告文件保存在 AutoDL 数据盘，本仓库只保留结果摘要。
