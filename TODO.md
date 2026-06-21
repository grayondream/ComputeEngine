# ComputeEngine 项目进度

> 最后更新: 2026-06-20

---

## 架构修订记录 (基于 Review 反馈)

| # | 修订内容 | 状态 |
|---|---------|------|
| 1 | **Backend 接口职责分离** — 拆为 Device/Allocator/StreamManager/KernelExecutor 四个子接口 | ✅ 已更新文档 |
| 2 | **Tensor Storage 模型** — 不再直接持有内存，改为 `shared_ptr<Storage>` + offset，支持零拷贝 View/Slice/Reshape | ✅ 已更新文档 |
| 3 | **Graph Compile → ExecutionPlan** — 不再对每个 Node 调用 ExecuteOp，先 Compile 生成 ExecutionPlan 再 Execute | ✅ 已更新文档 |
| 4 | **MemoryPool 双级策略** — <1MB 用 power-of-two bucket, ≥1MB 用 Best-Fit + Buddy System | ✅ 已更新文档 |
| 5 | **Unified Memory 谨慎处理** — 取消自动透明 Swap, 改为 Scheduler/KV Manager 显式控制 | ✅ 已更新文档 |
| 6 | **ResourceTracer 简化** — v1 只保留 TrackAlloc/TrackFree/DumpLeaks 三个 API, 移除 UUID 复杂性 | ✅ 已更新文档 |
| 7 | **Op 系统分层** — 拆为 OpSchema(元信息+形状推导) + KernelRegistry(后端Kernel管理) | ✅ 已更新文档 |
| 8 | **Backend 取消 ExecuteOp** — 算子路由由 KernelRegistry 负责，Backend 只负责 Launch/Alloc/Copy/Compile | ✅ 已更新文档 |
| 9 | **异步执行模型** — 新增 StreamManager::RecordEvent/WaitEvent, 支持跨流依赖和 CUDA Graph | ✅ 已更新文档 |
| 10 | **Compile Layer 引入** — Graph → Compile → ExecutionPlan → Execute 四阶段流水线 | ✅ 已更新文档 |
| 11 | **Document consistency cleanup** — UML 图与实现代码块统一 (Storage Allocator 类型, MemoryPool Deallocate 签名) | ✅ 已更新文档 |
| 12 | **ExecutionPlan 缓存键** — 增加 graph_hash / backend_version / compile_flags | ✅ 已更新文档 |
| 13 | **动态 Shape 策略** — Memory Planning 增加静态/半动态/完全动态三层策略 + ShapeSignature 缓存 | ✅ 已更新文档 |
| 14 | **Tensor Layout 枚举** — 增加 Layout::NCHW/NHWC/BLOCKED/CUSTOM, 区分逻辑 shape 与物理布局 | ✅ 已更新文档 |
| 15 | **KernelRegistry 细粒度缓存** — 缓存 Key 扩展为 op_type + device + dtype + shape_hash + compile_options | ✅ 已更新文档 |
| 16 | **资源约束感知调度** — 异步执行模型增加 ResourceBudget, 避免并行节点 OOM | ✅ 已更新文档 |
| 17 | **Reference 统一基准强化** — 明确 Reference(CPU Scalar) → 各后端分层容差验证 | ✅ 已更新文档 |
| 18 | **Engine 类定义** — 补全 API 示例中缺失的 LoadModel/CreateSession/CreateLLMEngine 接口 | ✅ 已更新文档 |

---

## 总体进度

```
Phase 1          Phase 2          Phase 3          Phase 4          Phase 5          Phase 6          Phase 7
Core+CPU+Ref     Graph+Runtime    CUDA Backend     OpenCL+Vulkan    LLM Engine       Quantize+Load    CI+Deploy
  (8 wk)           (5 wk)          (6 wk)           (6 wk)          (6 wk)           (5 wk)           (4 wk)
──────●──────────────●──────────────●──────────────●──────────────●──────────────●──────────────●──→
     W1             W9             W14             W20             W26             W32             W37
```

---

## Phase 1: Core 基础 + CPU Backend + Reference Ops (W1–W8)

| 子阶段 | 内容 | 状态 | 进度 | 前置 | 预计工期 |
|--------|------|------|------|------|---------|
| **1.1** CMake 构建系统 | `CMakeLists.txt`, `cmake/CompilerOptions.cmake`, `cmake/FindBackends.cmake` | ✅ 已完成 | 100% | — | 3 天 |
| **1.2** Core 基础类型 | `status.h`, `dtype.h`, `shape.h`, `half.h`, `device_type.h`, `macros.h` | ✅ 已完成 | 100% | 1.1 | 5 天 |
| **1.3** Tensor Storage 模型 | `tensor.h/.cpp`, Storage + stride-based 零拷贝 View/Slice/Reshape/Transpose | ✅ 已完成 | 100% | 1.2 | 5 天 |
| **1.4** Platform 基础设施 | `logger.h/.cpp`, `timer.h/.cpp`, `thread_pool.h/.cpp`, `profiler.h/.cpp` | ✅ 已完成 | 100% | 1.1 | 4 天 |
| **1.5** Backend 接口 + Registry | `backend.h` (4 子接口), `backend_registry.h/.cpp`, `REGISTER_BACKEND` 宏 | ✅ 已完成 | 100% | 1.2 | 4 天 |
| **1.6** CPU Backend 框架 | `cpu_backend.h/.cpp`, `cpu_context.h`, `dispatch.h/.cpp` (cpuid 检测) | ✅ 已完成 | 100% | 1.5 | 3 天 |
| **1.7** CPU MatMul Kernel | `matmul_scalar.cpp` → `matmul_avx.cpp` → `matmul_neon.cpp` | ✅ 已完成 | 100% | 1.6 | 5 天 |
| **1.8** CPU Conv2D Kernel | `conv_scalar.cpp`, `conv_avx.cpp`, `conv_neon.cpp` (im2col+GEMM, Winograd) | ✅ 已完成 | 50% | 1.7 | 5 天 |
| **1.9** CPU Norm Kernel | `norm_scalar.cpp`, `norm_avx.cpp`, `norm_neon.cpp` (LayerNorm, RMSNorm, BatchNorm) | ✅ 已完成 | 70% | 1.7 | 3 天 |
| **1.10** CPU Activation Kernel | `activation_scalar.cpp`, `activation_avx.cpp`, `activation_neon.cpp` | ✅ 已完成 | 80% | 1.7 | 3 天 |
| **1.11** CPU Attention Kernel | `attention_avx.cpp`, `attention_neon.cpp` (SDPA, causal, GQA/MQA) | ⬜ 待实现 | 0% | 1.7 | 4 天 |
| **1.12** Op 系统 | `op.h`, `op_registry.h/.cpp` (OpSchema + KernelRegistry) | ✅ 已完成 | 100% | 1.5 | 3 天 |
| **1.13** Reference Ops | `ref_ops.h/.cpp` (MatMul, Conv, Norm, Activation, Attention, RoPE) | ✅ 已完成 | 100% | 1.12 | 5 天 |
| **1.14** Memory 模块 | `allocator.h/.cpp`, `memory_pool.h/.cpp`, `unified_memory.h/.cpp` | ✅ 已完成 | 100% | 1.5 | 4 天 |
| **1.15** Resource Tracing v1 | `resource_tracer.h/.cpp`, `TracingAllocator` | ✅ 已完成 | 100% | 1.14 | 2 天 |
| **1.16** Testbed + 单元测试 | `test_runner.h`, `test_core.cpp`, `test_tensor.cpp`, `test_backend*.cpp`, `test_ops.cpp`, `test_memory.cpp`, `test_resource_tracer.cpp`, `test_platform.cpp` | ✅ 设计完成 | 0% | 1.13 | 6 天 |

**Phase 1 里程碑**: `./testbed --unit --backend cpu` 全部通过, CPU 后端与 Reference 数值一致 (atol≤1e-5)
**预计完成**: W8

---

## Phase 2: 计算图 + Session + Runtime (W9–W13)

| 子阶段 | 内容 | 状态 | 进度 | 前置 | 预计工期 |
|--------|------|------|------|------|---------|
| **2.1** Graph IR 数据结构 | `graph.h/.cpp` (GraphDef, NodeDef, EdgeDef, Attribute), 拓扑排序, 序列化 | ✅ 设计完成 | 0% | Phase 1 | 4 天 |
| **2.2** Graph Optimizer Pass | `graph_optimizer.h/.cpp` (ConstFold, DeadCodeElim, OpFusion, LayoutTransform, MemoryPlanning) | ✅ 设计完成 | 0% | 2.1 | 5 天 |
| **2.3** Graph Compiler | `execution_plan.h` (KernelNode, MemorySchedule, Dependency DAG, ShapeSignature), ExecutionPlanCache | ✅ 设计完成 | 0% | 2.2 | 5 天 |
| **2.4** GraphExecutor | `graph_executor.h/.cpp` (Compile + Execute), CUDA Graph capture, 异步执行 | ✅ 设计完成 | 0% | 2.3 | 4 天 |
| **2.5** Session + Pipeline | `session.h/.cpp`, `pipeline.h/.cpp` | ✅ 设计完成 | 0% | 2.4 | 3 天 |
| **2.6** ONNX Parser | `onnx_parser.h/.cpp` (ProtoBuf → GraphIR, 权重提取) | ✅ 设计完成 | 0% | 2.1 | 5 天 |
| **2.7** Model Registry | `model.h/.cpp`, `model_registry.h/.cpp` | ✅ 设计完成 | 0% | 2.6 | 2 天 |
| **2.8** 集成测试 | `test_graph.cpp`, `test_runtime.cpp` | ✅ 设计完成 | 0% | 2.7 | 4 天 |

**Phase 2 里程碑**: ONNX → Graph → Optimize → Execute 全链路通过
**预计完成**: W13

---

## Phase 3: CUDA Backend (W14–W19)

| 子阶段 | 内容 | 状态 | 进度 | 前置 | 预计工期 |
|--------|------|------|------|------|---------|
| **3.1** CUDAContext RAII | `cuda_context.h/.cpp` (cublas/cudnn handle, 多流, Events) | ✅ 设计完成 | 0% | Phase 1 | 3 天 |
| **3.2** CUDAMemory + Pool | `cuda_memory.h/.cpp` (cudaMallocAsync, Pinned Memory) | ✅ 设计完成 | 0% | 3.1 | 3 天 |
| **3.3** CUDABackend 集成 | `cuda_backend.h/.cpp` (四子接口实现) | ✅ 设计完成 | 0% | 3.1+3.2 | 3 天 |
| **3.4** CUDA GEMM | `gemm.cu` (cuBLAS, FP32/FP16/TF32/INT8) | ⬜ 待实现 | 0% | 3.3 | 4 天 |
| **3.5** CUDA Conv | cuDNN Conv wrapper | ⬜ 待实现 | 0% | 3.3 | 3 天 |
| **3.6** CUDA FlashAttention | `attention.cu` (FlashAttention-2, causal, ALiBi, GQA) | ⬜ 待实现 | 0% | 3.3 | 5 天 |
| **3.7** CUDA Fused Kernels | `norm.cu`, `activation.cu` (fused LayerNorm/GELU/SiLU) | ⬜ 待实现 | 0% | 3.3 | 4 天 |
| **3.8** CUDA Graph 集成 | BeginGraphCapture/EndGraphCapture/LaunchGraph | ⬜ 待实现 | 0% | 3.3 | 3 天 |
| **3.9** CC 分派 | 按 Compute Capability 选择最优 kernel 路径 | ⬜ 待实现 | 0% | 3.4 | 2 天 |
| **3.10** CUDA 测试 | `test_backend_cuda.cpp`, Reference 验证 (atol≤1e-4) | ⬜ 待实现 | 0% | 3.9 | 5 天 |

**Phase 3 里程碑**: CUDA 后端通过全部 Reference 验证, FlashAttention 正确
**预计完成**: W19

---

## Phase 4: OpenCL + Vulkan Backend (W20–W25)

| 子阶段 | 内容 | 状态 | 进度 | 前置 | 预计工期 |
|--------|------|------|------|------|---------|
| **4.1** OpenCL Context + Compiler | `cl_context.h/.cpp`, `cl_kernel_compiler.h/.cpp` (编译+缓存) | ✅ 设计完成 | 0% | Phase 1 | 4 天 |
| **4.2** OpenCL Backend 集成 | `opencl_backend.h/.cpp` (四子接口) | ✅ 设计完成 | 0% | 4.1 | 3 天 |
| **4.3** OpenCL Kernels | `matmul.cl`, `conv.cl`, `norm.cl`, `attention.cl` | ⬜ 待实现 | 0% | 4.2 | 6 天 |
| **4.4** OpenCL 移动端适配 | Adreno/Mali/PowerVR 调优 | ⬜ 待实现 | 0% | 4.3 | 4 天 |
| **4.5** Vulkan Context + Pipeline | `vk_context.h/.cpp`, `vk_compute_pipeline.h/.cpp` | ✅ 设计完成 | 0% | Phase 1 | 5 天 |
| **4.6** Vulkan Backend 集成 | `vulkan_backend.h/.cpp` (四子接口) | ✅ 设计完成 | 0% | 4.5 | 3 天 |
| **4.7** Vulkan Shaders | `matmul.comp`, `conv.comp`, `attention.comp` | ⬜ 待实现 | 0% | 4.6 | 5 天 |
| **4.8** 跨后端验证 | `test_backend_opencl.cpp`, `test_backend_vulkan.cpp` | ⬜ 待实现 | 0% | 4.4+4.7 | 5 天 |

**Phase 4 里程碑**: OpenCL/Vulkan 通过验证, 移动端实测通过
**预计完成**: W25

---

## Phase 5: LLM 推理引擎 (W26–W31)

| 子阶段 | 内容 | 状态 | 进度 | 前置 | 预计工期 |
|--------|------|------|------|------|---------|
| **5.1** Tokenizer | `tokenizer.h/.cpp` (BPE + SentencePiece) | ⬜ 待设计 | 0% | Phase 1 | 5 天 |
| **5.2** KV Cache 基础 | `kv_cache.h/.cpp` (存取, 多序列, CoW) | ✅ 设计完成 | 0% | Phase 3 | 4 天 |
| **5.3** PagedAttention | `paged_attention.h/.cpp` (Block Table, Prefill/Decode, Prefix Cache) | ✅ 设计完成 | 0% | 5.2 | 6 天 |
| **5.4** Attention 后端适配 | 各后端 Attention (FlashAttn/tile/subgroup/vectorized) | ✅ 设计完成 | 0% | 5.3 | 4 天 |
| **5.5** Continuous Batching | `scheduler.h/.cpp` (Waiting/Running/Paused, 优先级, eviction) | ✅ 设计完成 | 0% | 5.3 | 5 天 |
| **5.6** Sampler | `sampler.h/.cpp` (top-p, top-k, temperature, repetition penalty) | ✅ 设计完成 | 0% | 5.1 | 3 天 |
| **5.7** LLM Engine 编排 | `llm_engine.h/.cpp` (Generate → Chat → ContinuousBatch) | ✅ 设计完成 | 0% | 5.1-5.6 | 5 天 |
| **5.8** GGUF Parser | `gguf_parser.h/.cpp` (Header, Tensor Info, Quantized Weight) | ✅ 设计完成 | 0% | 5.7 | 4 天 |
| **5.9** LLM 端到端测试 | `test_llm.cpp` (llama.gguf → Prefill → Decode → 输出) | ⬜ 待实现 | 0% | 5.8 | 4 天 |

**Phase 5 里程碑**: llama-3.1-8B.gguf 端到端推理, Continuous Batching 正常
**预计完成**: W31

---

## Phase 6: 量化 + 模型加载 (W32–W36)

| 子阶段 | 内容 | 状态 | 进度 | 前置 | 预计工期 |
|--------|------|------|------|------|---------|
| **6.1** INT8 量化 (W8A8) | Per-tensor/channel 校准, 量化/反量化 kernel, INT8 GEMM | ⬜ 待设计 | 0% | Phase 1 | 5 天 |
| **6.2** INT4 量化 (W4A16) | AWQ/GPTQ 权重打包, INT4→FP16 解量化 kernel | ⬜ 待设计 | 0% | 6.1 | 5 天 |
| **6.3** SmoothQuant (W8A16) | 激活量化 + 权重量化分离 | ⬜ 待设计 | 0% | 6.1 | 3 天 |
| **6.4** 量化 Kernel 全后端 | CPU/CUDA/OpenCL/Vulkan INT8/INT4 kernel | ⬜ 待实现 | 0% | 6.2+6.3 | 6 天 |
| **6.5** Graph Optimizer 量化 Pass | QuantizeInsert Pass, 精度分析 | ✅ 设计完成 | 0% | 6.4 | 3 天 |
| **6.6** 量化测试 | `test_quant.cpp`, INT4/INT8 精度验证 | ⬜ 待实现 | 0% | 6.5 | 4 天 |

**Phase 6 里程碑**: INT4 llama-3.1-8B 内存减半, perplexity 损失 <1%
**预计完成**: W36

---

## Phase 7: Testbed + CI + 部署 (W37–W40)

| 子阶段 | 内容 | 状态 | 进度 | 前置 | 预计工期 |
|--------|------|------|------|------|---------|
| **7.1** Testbed CLI | 命令行参数解析 (--all/--unit/--backend/--filter/--benchmark) | ✅ 设计完成 | 0% | Phase 1 | 3 天 |
| **7.2** Benchmark 套件 | `benchmark.cpp` (MatMul/Conv/Attention/LLM 延迟, 后端对比) | ✅ 设计完成 | 0% | 7.1 | 4 天 |
| **7.3** CI/CD Pipeline | GitHub Actions (构建, 测试, 覆盖率, 多后端矩阵) | ⬜ 待实现 | 0% | 7.1 | 4 天 |
| **7.4** 覆盖率达标 | core/100%, backend/90%, graph/85%, op/95%, memory/90%, llm/80% | ⬜ 待实现 | 0% | 7.3 | 3 天 |
| **7.5** API 文档 | Doxygen 配置, API Reference 自动生成 | ⬜ 待生成 | 0% | 7.4 | 3 天 |
| **7.6** 编译/移植指南 | `docs/build.md`, `docs/porting.md` | ⬜ 待生成 | 0% | 7.5 | 3 天 |
| **7.7** 打包发布 | CMake install + CPack, 版本号, CHANGELOG | ⬜ 待实现 | 0% | 7.6 | 2 天 |

**Phase 7 里程碑**: CI 全绿, 覆盖率达标, v1.0 发布
**预计完成**: W40

---

## 测试进度

| 测试文件 | 阶段 | 状态 | 覆盖率目标 |
|---------|------|------|-----------|
| `testbed/tests/test_core.cpp` | 1.16 | ⬜ 待实现 | 100% Status/DType/Shape/half |
| `testbed/tests/test_tensor.cpp` | 1.16 | ⬜ 待实现 | 100% Tensor |
| `testbed/tests/test_backend.cpp` | 1.16 | ⬜ 待实现 | 90% Backend |
| `testbed/tests/test_backend_cpu.cpp` | 1.16 | ⬜ 待实现 | 90% CPU Backend |
| `testbed/tests/test_ops.cpp` | 1.16 | ⬜ 待实现 | 100% Ops vs Reference |
| `testbed/tests/test_memory.cpp` | 1.16 | ⬜ 待实现 | 90% Memory |
| `testbed/tests/test_resource_tracer.cpp` | 1.16 | ⬜ 待实现 | 90% Tracer (v1) |
| `testbed/tests/test_platform.cpp` | 1.16 | ⬜ 待实现 | 90% Platform |
| `testbed/tests/test_graph.cpp` | 2.8 | ⬜ 待实现 | 80% Graph |
| `testbed/tests/test_runtime.cpp` | 2.8 | ⬜ 待实现 | 85% Runtime |
| `testbed/tests/test_backend_cuda.cpp` | 3.10 | ⬜ 待实现 | 90% CUDA Backend |
| `testbed/tests/test_backend_opencl.cpp` | 4.8 | ⬜ 待实现 | 90% OpenCL Backend |
| `testbed/tests/test_backend_vulkan.cpp` | 4.8 | ⬜ 待实现 | 90% Vulkan Backend |
| `testbed/tests/test_llm.cpp` | 5.9 | ⬜ 待实现 | 80% LLM |
| `testbed/tests/test_quant.cpp` | 6.6 | ⬜ 待实现 | 85% Quantization |
| `testbed/tests/benchmark.cpp` | 7.2 | ⬜ 待实现 | — |

---

## 文档进度

| 文档 | 状态 | 说明 |
|------|------|------|
| `ARCHITECTURE.md` | ✅ 完成 (4567 行) | 完整架构设计 + Mermaid 图 + 模块接口 + 细化开发计划 |
| `TODO.md` | ✅ 完成 | 本文件 |
| API 参考文档 | ⬜ 待生成 | Doxygen 自动化 |
| 编译指南 | ⬜ 待生成 | CMake 选项说明 |
| 移植指南 | ⬜ 待生成 | 新增后端/算子/量化方法 |

---

## 近期任务 (Next Steps — Phase 1 前 5 个子阶段)

1. [x] **1.1** CMake 构建系统搭建 (3 天)
2. [x] **1.2** Core 基础类型实现 — Status, DType, Shape, half, DeviceType (5 天)
3. [x] **1.3** Tensor Storage 模型实现 — stride-based 零拷贝 View/Slice/Reshape (5 天)
4. [x] **1.4** Platform 基础设施 — Logger, Timer, ThreadPool, Profiler (4 天)
5. [x] **1.5** Backend 接口 + BackendRegistry — 4 子接口 + 注册宏 (4 天)
6. [ ] **1.7** CPU MatMul Kernel — scalar + AVX + NEON tiled (5 天)
7. [ ] **1.12** Op 系统 — OpSchema + KernelRegistry (3 天)
