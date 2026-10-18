# P3 · GPU 后端(CUDA 或 Vulkan)+ GPU 硬件探测

> 设计来源:`docs/readme.md` §5.3(CUDA)、§5.4(Vulkan)、§2(硬件探测)、§18(决策)。GPU 后端选型为决策点,候选见 §8(两后端都实现 / 运行期激活其一)。

## 1. 目标与验收(DoD)

- **目标**:激活单一 GPU 后端(CUDA 或 Vulkan),GPU 能力位探测完整、能力位驱动 kernel 选择;同图 CPU/GPU 结果一致。
- **验收标准**:
  1. `DeviceSelector::select()` 返回实际选中的 `GpuBackend::{CUDA,VULKAN}`,配置可强制指定。
  2. GPU 后端能把 P1 的通用图(MatMul/逐元素)编译并执行;结果与 CPU oracle 一致(容差)。
  3. **同图 CPU/GPU 一致**:同一 IR 图在 CPU 与 GPU 后端执行结果数值一致(容差内)。
  4. GPU 硬件画像完整:探测 `supports_tensor_core / fp16/bf16/tf32/int8/int4/fp8/fp4_mma / sparsity`、内存带宽、TMA/bulk copy(能力位)等(§2.1+§5.3 组合)。

## 2. 范围与边界

- **范围内**:GPU 后端基础设施(init/shutdown/alloc/free/copy_D2H/H2D/D2D/execute_async/wait)、能力位探测、命令式 kernel(MatMul/逐元素起步)、CPU↔GPU 传输。
- **范围外**:调度/混合(P4+/P9+)、注意力/LLM(P6+)、GPU 专属融合 pass、FP8/TMA 深度路径(§5.3 高级组合可后置)。

## 3. 需实现/落地的模块

- CUDA:
  - `backends/cuda/CudaBackend`、`core/hardware/HardwareDetector::detect_cuda(device)`、`backends/cuda/kernels/`
  - 能力位驱动策略矩阵(§5.3:`supports_fp8_mma && supports_bulk_tensor_copy` → TMA+FP8 路径…)初版聚焦通用路径,高级矩阵后置
- Vulkan:
  - `backends/vulkan/VulkanBackend`、GLSL→SPIR-V→`vkCmdDispatch`、`detect_vulkan(device)`
  - VMA + staging buffer;`VK_KHR_cooperative_matrix / VK_KHR_copy_commands2` 能力位检测(§5.4)

## 4. 接口要点

- 全部复用 `Backend`(§3.3)与 `BackendCaps`(§3.7):`compile(Graph&)`、`execute_async(BatchPlan, ExecutionContext&)`、`wait(Timeout)`。
- `TransferManager.register_route / transfer / prefetch_async / wait`(§7.5)打通 CPU↔GPU 两条传输路径。
- GPU 硬件画像进 `HardwareProfile`,与 CPU 统一框架(§18)。

## 5. 实现要点

- **正确性**:GPU kernel 以 CPU oracle 为准,逐算子 TDR-less 比对(§19 数值一致性)。
- **同步首制**:MVP 可 `execute`(同步)+ `wait` 收敛;真正的异步流/图捕获留 P10(Event 同步)。
- **传输**:`copy_to_device/copy_to_host/copy_d2d` 先同步,CUDA 用 `cudaMemcpy*`,Vulkan 用 staging + `vkCmdCopyBuffer`(§7.5 矩阵)。
- **能力位默认回退**:探测失败 → 保守标量路径(§19)。

## 6. 测试与验证

- CPU vs GPU 后端同图一致性测试(P1 图集)。
- 传输往返(D2H/H2D/D2D)数据完整性 + 大 buffer。
- GPU 能力位探测单元测试(注入/断言已知能力位)。
- 每 kernel 的数值 + 正确性;性能 sanity(相对 CPU 的有意义起点即可,不做硬阈值)。

## 7. 依赖

- P1(P2 传统算子可用但非必需;本阶段侧重打通后端通用路径)。
- 选型依赖决定:用户先裁定 **CUDA 还是 Vulkan**;CUDA 需要 NVCC/CUDA 工具链与设备;Vulkan 需要 Vulkan-SDK/VMA/Vulkan-Hpp(§20.4)。

## 8. 决策点(默认建议 · 待实现时裁定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现 P3 时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| GPU 后端选型 | **CUDA 且 Vulkan(两者都实现)候选** | 见下方说明:破"只能二选一"的构建约束(两后端都构建),但运行期仍只激活其一 |
| 高级组合路径(§5.3) | 先通用,高级延后 | FP8/TMA/FP4/MXFP 矩阵路径留到需要时 |

### 「两个后端都实现」的说明与设计对齐(如采纳)
- 设计 §1 设备拓扑是"单一 GPU 后端(CUDA 或 Vulkan)",§18 决策"二选一以求简"。该候选建议两者都开发。
- 协调:**构建期同时产出** CudaBackend 与 VulkanBackend(各自独立静态库,§19);**运行期仍只激活一个**(DeviceSelector 按配置/可用性),保持"GPU 侧单活跃后端"简拓扑。
- 影响:额外一整套后端(+shader/vulkan)工程量;BackendMatrix/TransferManager 需同时补两条传输路径(§7.5)。建议实现时**先 CUDA、后 Vulkan**,各自过"同图 CPU/GPU 一致"验收。
- 若未来要"双 GPU 后端同时活跃",需单独修订 §1 设备拓扑约束——本 Spec 不做此假设。