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
## 9. 实现期裁定(2026-11-08 回写)

> 实现提交范围 `7ce453c..56ac645`(Task 1–6)。本节覆盖并取代 §8 的默认建议;§1–§7 的 DoD/接口不作删改。最终验证数字见 `todo.md` / `progress.md`。

| 决策点 | 实现期裁定 | 说明 |
|---|---|---|
| 基准框架 | **不引入 Google Benchmark**;仓库内 `ie_bench`(`std::chrono`)+ Release 性能 sanity | 避免新增依赖与 CI 噪声;`ie_bench` 输出 profile、选中变体与 GFLOPS;`MatMulBenchmarkSanity.Avx2BeatsScalarOnLargeMatrix` 仅 Release 运行并断言加速比,Debug skip |
| SIMD 编译与分派 | **函数级 `__attribute__((target(...)))` + 运行时能力位分派**;不使用全局 `-march` | 基线二进制保持可移植;`sse2` / `avx2+fma` / `avx512f+bw+vl` 三级变体由 `CpuKernelCatalog::select` 按能力位选择;尾部走标量 |
| AVX-512 验证 | 本机(Zen3,AMD Ryzen 5 5600X)无 AVX-512:**数值/性能用例 skip** | AVX-512 内核仍参与编译,并用合成 `HardwareProfile` 验证能力位分派与目录选择;真机数值/吞吐待有 AVX-512 的机器补跑 |
| MLP 偏置 | **折叠进增广权重**(齐次坐标),不扩展 `Add` 的广播语义 | 输入/隐层各追加一列常量 1;`W1_aug [9,17]`、`W2_aug [17,9]`、`W3_aug [9,4]`;ReLU 精确保留该列 1,P0 `Add` 同 shape 契约不变 |
| PyTorch 参考数据 | **一次性生成并提交** `tests/data/mlp_reference.h`;测试离线运行 | `tests/reference/gen_mlp_reference.py`(PyTorch 2.15、seed `20261106`、F32、9 位小数);重跑哈希一致;运行期不依赖 Python/PyTorch |
| 线程模型 | **持久 `ThreadPool`**,MatMul 按输出行切分,归约顺序固定 | `CpuBackend` 持有线程池(线程数 `cores*threads_per_core` 钳制 `[1,64]`);`parallel_for(m,1,...)` 每行独立、k 升序累加;`threads<=1` 调用线程内联;非重入 |
| CPU 分配器 | **`CpuAllocator` 64 字节对齐**;Engine 既有 `MallocAllocator` 路径保留 | `posix_memalign`(Windows `_aligned_malloc`);按 64 倍数统计;重复释放/null 安全;`CpuBackend::execute_async` 仍优先使用调用方 `context.allocator` |
| 能力位命名 | 沿用规范名 **`avx512vnni` / `avx512bf16` / `amx-*`** | `amx-tile`/`amx-int8`/`amx-bf16`/`amx-fp16`;与 `KernelVariant.req.required_caps` 及 profile 映射一致,不写架构名 if-else 链 |
| VNNI/AMX 内核 | P1 **仅登记能力位,不提供 F32 内核** | VNNI/AMX 的收益面向后续 INT8/BF16 dtype 里程碑;P1 变体目录止于 `avx512` |
| Cost model | **按 `estimated_tflops` 取最高**;实测微调留 P7 | 四档成本固定为 1/2/4/8;`select_best_cpu_variant` 只做优先级排序,不做运行期标定 |

### 9.1 MLP 增广维度修正(17/9)

- 原 brief 的 `h1r=[4,16]` 与 `w2=[17,8]` 本身就互相矛盾:偏置折叠后,隐层输出必须带上常量列,`h1r` 应为 `[4,17]`;`w2` 的输入维必须是 `17`(16 个隐单元 + 1 个常量),输出维应为 `9`(8 个隐单元 + 1 个常量)。按原 shape 会触发 `MatMul inner dimensions mismatch`,且第三层偏置无法折叠。
- 实现取 `x[4,9] → W1_aug[9,17] → h1r[4,17] → W2_aug[17,9] → h2r[4,9] → W3_aug[9,4] → logits[4,4] → Softmax`;ReLU 对常量列做 `max(1,0)=1`,因此后续层仍可直接使用齐次坐标。黄金数据按同一前向计算,数值对齐验证了该修正。

### 9.2 Release `-Wmaybe-uninitialized` 修复

- GCC 15 在 Release 下对 `std::optional<Graph>` 的析构路径报 `-Wmaybe-uninitialized`(析构就地存储的 `Graph` 时会实例化其 `std::shared_ptr<Graph> backward_graph` 成员)。
- `CpuBackend::graph_` 改为 `std::unique_ptr<Graph>`(`std::make_unique<Graph>(graph)`),语义与生命周期不变,消除该告警;全局 `-Wall -Wextra -Wpedantic` 的全目标 Release 构建零告警。
