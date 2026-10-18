# P1 · CPU 后端完善 + 硬件能力探测

> 设计来源:`docs/readme.md` §2(硬件画像/探测器/能力驱动)、§5.2(CPU 后端)、§14(借鉴优化项)。本 Spec 是编码依据。

## 1. 目标与验收(DoD)

- **目标**:CPU 后端从"直通标量"升级为"能力位驱动的正确实现";硬件探测完整覆盖 CPU 能力位;小 MLP 数值对齐 PyTorch。
- **验收标准**:
  1. **小 MLP 对齐 PyTorch**:定义一个小 MLP(若干 Linear+ReLU+Softmax,固定 seed),本引擎输出与 PyTorch 参考输出 rtol/atol 容差内一致。
  2. CPU 能力位探测平台化:cpuid/`__builtin_cpu_supports` 探测 SSE2/AVX/AVX2/AVX512F+BW+VL/VNNI/BF16/FP16/AMX,ARM 侧 NEON/SVE/SVE2/SME 探测(¶20.2)。
  3. MatMul 有至少标量 + 一种 vector/矩阵加速变体,且按能力位自动选择。
  4. 新增 AVX2/AVX-512 kernel 通过数值与性能 sanity 测试。

## 2. 范围与边界

- **范围内**:CPU 能力位探测、MatMul/逐元素算子的 SIMD 变体、KernelVariant 选择打通、HardwareProfile 填充、OptimizationPlanner 产生首个 Plan。
- **范围外**:(本阶段)Conv/Pool/BN 真实实现(P2);GPU 后端(P3);规则/Transformer(P5+)。

## 3. 需实现/落地的模块

- `core/hardware/HardwareDetector`(CPU 路径,§2.2)
- `core/hardware/OptimizationPlanner`(首个能力驱动决策,§2.3)
- `core/tuning/KernelVariant + KernelVariantRegistry.select`(§6.2)
- `ops/common/MatMul + Relu + Softmax` 的 SIMD 变体
- `backends/cpu/CpuBackend` 完善(线程池 + CPU 分配器,§5.2)
- `backends/cpu/simd/`:AVX2 / AVX-512 / VNNI / AMX 的封装(按能力位条件编译)
- `backends/cpu/kernels/`:标量 + SIMD kernel 变体

## 4. 接口要点

- `HardwareDetector::detect_cpu()` → 填满 §2.1 `HardwareProfile`(计算单元/向量与矩阵加速/内存/数据搬运/执行特性/互连/部署)。
- `KernelVariant.req`:`required_caps`、`min_tensor_to_vector_ratio` 等前置条件;`Registry.select(op, dtype, shape, profile, kind)`。
- **能力位归一化**(§2、§6):把 CPU 的 AVX2/AVX-512/VNNI/AMX 差异表达成统一能力位,不写架构名 if-else 链。

## 5. 实现要点

- **正确性基准**:保持 CPU 为 oracle;SIMD kernel 必须与标量 oracle 在容差内逐算子比对(§19 风险对策)。
- **线程池**:沿用 P0 ThreadPool;MatMul 等可并行算子接入,保证结果确定(归约顺序固定)。
- **特化优先级**(§5.2 表):MatMul 依能力位走 `scalar → sse2 → avx2 → avx512 → avx512_vnni → amx_*`。
- **naive 先行**:先保证对,再加 SIMD;profiler 后再优化(§19)。

## 6. 测试与验证

- 能力位探测在不同机器上单元测试(注入假 CPUID 或用可探测子集断言一致性)。
- MatMul 标量 vs SIMD 全 shape 网格 io一致性;AVX2/AVX512 变体各自存在且有对应能力位分支。
- 对齐测试:小 MLP vs PyTorch(用 `onnxruntime` 或 numpy 参考,容差 `atol=1e-5, rtol=1e-3` 起步)。
- benchmark:同一 shape 下 SIMD vs 标量吞吐提升为有梯度;性能 sanity 加 CI 上限防止回归。

## 7. 依赖

- P0/P0.5(核心接口冻结完毕、KernelVariant 接口就位)。
- 需要 PYTHON/PyTorch 作为参考对齐的工具链(实验/CI 可选)。

## 8. 决策点(默认建议 · 待实现时裁定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现 P1 时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| 基准框架 | **Google Benchmark**(P0 候选) | §6 性能 sanity 用它 |
| CPU 能力位探测深度 | **x86 全量 + ARM 基础** | x86:SSE2/AVX/AVX2/AVX512(含 BW/VL/VNNI/BF16/FP16)+ AMX(CPUID leaf7);ARM:NEON 实现 + SVE/SME 探测接口占位 |
| Cost model 首版 | **保守启发式** | KernelVariant 选择用"高能力位优先 + 阈值(如 ρ、带宽)",不做真实 cost;运行期实测微调整体留 P7 |