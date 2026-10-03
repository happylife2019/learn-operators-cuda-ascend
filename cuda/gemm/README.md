# GEMM 学习记录

每完成一个优化版本，就在独立目录保留该版本的源码和 `README.md`。新版本不覆盖旧版本；所有版本在同一张 GPU、相同矩阵尺寸、相同精度和相同计时范围下比较。

**保存位置是本地 Git 仓库。** 在本地写源码与记录并推送 GitHub；AutoDL 只拉取、编译和运行。把 AutoDL 的正确性结果、计时和 Profiling 指标抄回本地对应版本的 `README.md`，再从本地提交、推送。AutoDL 容器可能消失，不能把唯一的源码或实验记录留在那里。

| 版本 | 目录 | 正确性 | Kernel 耗时 | 吞吐量 | 记录状态 |
| --- | --- | --- | --- | --- | --- |
| Naive | [`navie_gemm/`](navie_gemm/) | PASS | 1.109 ms | 1936.43 GFLOPS | AutoDL 单次运行；源码路径待统一 |
| Shared Memory Tiled | [`shared_memory/`](shared_memory/) | PASS | 0.834 ms | 2576.35 GFLOPS | AutoDL 单次运行；相对 Naive 约 1.33×，待多轮复测 |
| Thread-Tiled Shared Memory | [`shared_memory_v2/`](shared_memory_v2/) | PASS | 0.671 ms | 3201.76 GFLOPS | AutoDL 单次普通运行；另有 [Nsight Systems 记录](shared_memory_v2/README.md) |

上表数字来自 RTX 3080 Ti 的运行截图，Naive 测于 2026-09-29，Shared Memory Tiled 测于 2026-09-30，Thread-Tiled 测于 2026-10-03。表中各版并非同一次受控对比；Naive 运行路径与仓库路径也尚未统一。因此表内时间和比值只是阶段性记录，不是正式加速比。

## 每版记录什么

1. **改动**：相比上一版，改了哪些访存、线程分工或计算方式。
2. **正确性**：测试的矩阵尺寸、精度、参考实现、误差阈值和 PASS/FAIL。
3. **环境**：GPU、CUDA Toolkit、编译命令、源码提交号。
4. **性能**：只计 Kernel 的时间；预热后多次运行，记中位数、GFLOPS，以及相对 Naive 的加速比。
5. **分析**：提出瓶颈假设，用 Profiling 数据验证；没有测量就写“未测”，不填估计值。

各版保留独立目录与实测说明；后续同条件对比时再补正式加速比。

## Profiling 放在哪一步

- **目前可用**：AutoDL 上的 Nsight Systems 可以核对 kernel 次数、启动配置、GPU 时间线以及数据拷贝。Profiler 中的时间与普通 CUDA Event 计时分开记录。
- **目前受限**：此 AutoDL 容器运行 Nsight Compute 返回 `ERR_NVGPUCTRPERM`，不能读取占用率、访存吞吐、stall 等硬件计数器。仅凭 Nsight Systems 时间线不推断这些瓶颈。
- **后续**：若获得性能计数器权限，再对照各版分析实际访存与计算瓶颈；在此之前先保持源码、正确性与可复现命令完整。
