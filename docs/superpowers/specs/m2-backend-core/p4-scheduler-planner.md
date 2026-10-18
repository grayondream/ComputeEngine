# P4 · BackendScheduler + OptimizationPlanner(能力驱动调度/放置)

> 设计来源:`docs/readme.md` §2.3(OptimizationPlanner)、§3.2(Scheduler 双模式)、§5.1(EngineConfig)、§7(HybridPlanner 雏形)。

## 1. 目标与验收(DoD)

- **目标**:实现"按能力画像给算子/子图分派到后端"与"跨设备放置+传输插入"的能力;CPU+GPU(有 GPU 时)可自动调度。
- **验收标准**:
  1. `OptimizationPlanner::plan()` 对给定图 + 硬件画像 + `ModelKind` 产出 `Plan`(启用 pass、首选 kernel、内存策略、量化策略、reason,§2.3)。
  2. `BackendScheduler` 能把 BatchPlan 内的算子分配到可用的后端(CPU 至少;有 GPU 则按能力位与 cost 分配)。
  3. 单设备回归:CPU-only 链路与 P2/P4 前结果一致;若有 GPU,CPU+GPU 自动切分图正确执行。
  4. 跨设备传输自动插入(`insert_transfers`),数据搬运正确不撕裂。
  5. `EngineConfig.hardware_adapt.enable_capability_driven_selection` 开关可关(回退保守路径)。

## 2. 范围与边界

- **范围内**:OptimizationPlanner 能力决策、BackendScheduler 分派、Node 放置决策(单机 CPU/GPU)、传输插入、`EngineConfig` 校验、`BackendMatrix` 能力矩阵(§5.5)。
- **范围外**:完整混合规划 P9+(专家/神经元/流水线/投机);运行期实时微调(P7);GPU 仅存在时的双设备协同细节(P4 先做"能分配、能传对")。

## 3. 需实现/落地的模块

- `core/scheduler/BackendScheduler`(§3.2/§1 调度层)
- `core/hardware/OptimizationPlanner`(能力决策,§2.3)
- `core/backend/DeviceSelector + BackendKind/BackendMatrix`(§5.1/§5.5)
- `core/hardware/CapabilityNormalizer`(跨厂商能力位归一,§15 目录)
- `hybrid/planner` 的**雏形**:`HybridPlanner::plan()` 最小可用(`analyze_op_types/classify_model_kind/decide_placement/insert_transfers`,§7.3),但不做负载均衡/成本模型

## 4. 接口要点

- `OptimizationPlanner::Plan`(§2.3);`Scheduler::BatchPlan`(§3.2 无 decode).
- `Node.placed_device`、`Node.kind` 标注参与放置。
- `TransferManager` 路由 CPU↔GPU(§7.5)。

## 5. 实现要点

- 决策只读能力画像(§18):GEMM 放 GPU 当 GPU 有 tensor core/带宽占优;否则留 CPU。
- 放置 `decide_placement` 首版用**保守规则**(能力位 + 简单 cost),不做贪心/RL(§20.6)。
- `insert_transfers` 只在真实跨设备依赖边插入;避免多余 copy_D2D。
- 后端能力矩阵防止把后端不支持的算子分给它(§5.5).

## 6. 测试与验证

- `OptimizationPlanner` 决策:给定不同硬件画像,断言 Plan 差异符合能力位差异;确定性与幂等。
- 跨设备切分正确性:CPU+GPU 执行同 CPU 全量 oracle 一致;传输点数据正确。
- 关闭能力驱动选择开关后回退一致。
- `BackendMatrix` 覆盖宿主后端算子。

## 7. 依赖

- P3(GPU 后端可用)、P1/P2(CPU 传统算子)。
- 若无 GPU 硬件,CI 用 CPU-only 断言 + GPU Plan 的注入式单测。

## 8. 决策点(默认建议 · 待实现时裁定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现 P4 时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| 放置算法 | **保守规则** | 能力位 + 简单 cost/memory 的保守规则;贪心/LP/RL 后置到 P15 优化里程碑 |
| Cost model | 保守成本(与放置同层的保守启发式) | 与 P1 的保守 Cost 一致;更精确 cost 后置 |
| 设备选择语义 | **配置优先 + 自动回退** | `EngineConfig.gpu_backend` 显式指定时优先;未指定时 `DeviceSelector` 自动探测可用后端并回退 CPU;能力位驱动选择开关可关 |