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

## 后续
- [ ] P0.5 接口冻结(见 `docs/superpowers/specs/m1-bootstrap/p0.5-interface-freeze.md`)
  - [x] Task 1 配置、特性门控、API/插件版本(`21dfa9a`)
  - [x] Task 2 编译层与调优接口(`a036089`,复审修复 `b31d1fa`)
  - [ ] Task 3–8
- [ ] 每个里程碑实现时:对照「决策点」节,从默认建议起步,裁定后回写 Spec/台账
- [ ] P0.5 重点复核:是否引入 MLIR、插件 ABI 提前锁定
