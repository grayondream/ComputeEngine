# TODO

## 当前任务(已完成)
- [x] 根据 `docs/readme.md` 设计文档,产出一整套按里程碑分文件夹的引擎实现 Spec(中文,面向实现者),置于 `docs/superpowers/specs/`(17 份文档)
- [x] 与用户澄清 Spec 范围/位置/语言/目录粒度等不确定点
- [x] 识别并整理全周期**决策点**及其默认建议,汇总到 `00-overview/decisions-log.md`
- [x] 创建并维护 `todo.md`、`progress.md`

## 决策处理方式(2026-10-18 修订 ◆)
- 不再在 Spec 期硬锁定决策;决策在对应里程碑实现时裁定并回写 Spec/台账。
- Spec 末节「决策点(默认建议 · 待实现时定)」为候选,实现者从默认建议起步。

## P0-MVP 实现(2026-10-31 完成 ✅)
- [x] 实现计划:`docs/superpowers/plans/2026-10-31-p0-mvp-skeleton.md`(10 个 Task)
- [x] 构建/工程:C++20、CMake + Ninja、系统 GoogleTest、`src/` 布局
- [x] `src/core/base`:Error/Status/Result(自研)、DType/Device/Layout/Shape
- [x] `src/core/ir`:Graph/Node/Value/Edge/ControlFlow、GraphBuilder、bind
- [x] `src/core/memory`:MallocAllocator、Tensor、MemoryPolicy、TransferManager 骨架
- [x] `src/core/backend`:Backend/BackendCaps/BackendRegistry/DeviceSelector/EngineConfig/BackendMatrix
- [x] `src/core/ops`:Op/OpRegistry、Add/Mul/MatMul CPU kernel
- [x] `src/core/hardware`:CapabilitySet/HardwareProfile、CPU 探测、OptimizationPlanner 骨架
- [x] `src/core/scheduler`:Scheduler/BatchPlan/StaticScheduler
- [x] `src/core/runtime`:ExecutionContext/Executor/Engine、CpuBackend
- [x] `src/core/plugin`、`src/core/tuning`、RNG 骨架
- [x] `src/demo/main.cpp`:`(a+b)*3` 输出 15,21,27
- [x] 验证:Debug/Release 各 49/49 ctest 通过
- [x] 回写 `p0-mvp.md` §8 与 `decisions-log.md` 的实现期裁定

## P0.5 接口冻结(2026-11-04 完成 ✅)
- [x] 实现计划:`docs/superpowers/plans/2026-11-04-p0.5-interface-freeze.md`(10 个 Task)
- [x] Task 1 配置、特性门控、API/插件版本(`21dfa9a`)
- [x] Task 2 编译层与调优接口(`a036089`,复审修复 `b31d1fa`)
- [x] Task 3 运行时执行器、传输与混合层(`155e8ab`,复审修复 `7a171e6`)
- [x] Task 4 规则推理层(`4129dae`)
- [x] Task 5 算子族与实验算子(`a43ff96`,测试补强 `877fe86`)
- [x] Task 6 LLM 层与 Tokenizer/Sampler/ModelLoader(`73634d3`)
- [x] Task 7 训练层(`7e3afd2`,默认值/字段补强 `7ca506b`)
- [x] Task 8 校验与调试层(`815dfd3`)
- [x] Task 9 模型转换层与量化插件(`63b18ca`,复审修复 `b28daf5`)
- [x] Task 10 C ABI、支撑层、接口清单与文档回写(`fa46124`)
- [x] 关键模块:feature flags/API 版本、pass/tuning、runtime/transfer/hybrid、rules、四族 ops、LLM、training、validation、converter/quant、C ABI、security/fault/observability/tools/sdk/packaging/benchmarks
- [x] 接口清单:`docs/superpowers/specs/m1-bootstrap/p0.5-interface-inventory.md`(79 个头文件 / 10 个契约组 / 冻结版本 `0.1`)
- [x] 验证:Debug **116/116**、Release **116/116** `ctest` 通过;`ie_demo` 仍输出 `result: 15, 21, 27`;P0-MVP 回归通过
- [x] 回写 `p0.5-interface-freeze.md` §8 与 `decisions-log.md` 实现期裁定
- [x] 重点复核已裁定:不引入 MLIR(自定义 `Graph::bind`)、插件 ABI 提前锁定 `0.1`、JSON 延后、`KernelVariant` 对齐 §6.2、`Status` 为规范 void 结果
- [x] 整分支复审修复:门控权威性、transfer/copy 语义、Sandbox、C ABI nothrow、转换/校验表驱动 DoD #1 覆盖、inventory 总数与文档同步

## P1 CPU 硬件(进行中)
- [x] Task 1 硬件探测平台化(`7ce453c`):`CpuFeatures`/`CpuTopology`/`build_cpu_profile`、x86 CPUID + `__builtin_cpu_supports`、ARM `getauxval`、Linux 拓扑探测、fingerprint 扩展
- [x] Task 2 CPU 分配器与确定性线程池(`ace983f`):`CpuAllocator`(64B 对齐、64 倍数统计、重复释放安全)、`ThreadPool`(确定性 chunk 分配、`threads<=1` 调用线程内联、非重入)
- [x] Task 3 SIMD 内核与能力目录(`3165aab`):`KernelFns`、12 个 CPU 变体(3 op × 4 档)、标量/SSE2/AVX2/AVX-512 内核、`CpuKernelCatalog::select`;复审补测(`11ebcd6`)覆盖 ReLU/Softmax 向量循环与 2-D softmax
- [x] Task 4 ReLU/Softmax 算子与 MatMul 分派(`7cd5a6f`):`ReluOp`/`SoftmaxOp`(1D/2D 稳定 softmax)、`ExecutionContext` 注入 `KernelFns*`/`ThreadPool*`、`Executor::run` 默认参数、MatMul 按行并行(阈值 32768)、`OptimizationPlanner` 能力驱动首选内核

## 后续
- [ ] 每个里程碑实现时:对照「决策点」节,从默认建议起步,裁定后回写 Spec/台账
- [ ] P1 起在 P0.5 冻结接口上增量实现 CPU/硬件探测等能力
