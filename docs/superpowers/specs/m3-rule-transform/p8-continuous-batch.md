# P8 · 连续批处理 + HardwareAwareCache(吞吐最大化)

> 设计来源:`docs/readme.md` §3.2(Scheduler)、§6.3(HardwareAwareCache)、§14(连续批 🟡中)、§20.8。

## 1. 目标与验收(DoD)

- **目标**:`ContinuousBatchScheduler` 让多条生成请求在迭代中共享批次,提升整体吞吐;调度决策落到 `BatchPlan`。
- **验收标准**:
  1. 多请求共享一次迭代执行:`BatchPlan.prefills + decodes` 被打包成一批;每请求维护独立 KV/位置状态。
  2. **吞吐提升**:并发 N 请求的端到端吞吐(idle-QPS/p99 口径)> 逐请求串行处理;并有基准报告(对比简单调度器)。
  3. 请求完成即被调出批次(`on_request_complete`),新请求在迭代边界加入;**不做抢占**也可达成本里程碑(§3.2 `preempt` 可留)。
  4. `HardwareAwareCache.get/put` 在迭代间复用 autotune 决策,重复 shape 不再重测。
  5. 批次不变性 sanity:含 batch 行为不改变单请求数值正确性。

## 2. 范围与边界

- **范围内**:ContinuousBatchScheduler、批内 KV 管理(结合 PagedKVCache)、迭代粒度调度、完成即出批。
- **范围外**:Chunked Prefill(§12.2 TODO 后继)、优先级/复杂抢占、PD 分离、跨请求前缀缓存命中率优化(可后置)、混合(传统+Transformer)调度(P14 编排)。

## 3. 需实现/落地的模块

- `core/scheduler/ContinuousBatchScheduler`(§3.2)
- `core/scheduler/BatchPlan` 编排(共享批次,decodes/prefills 分组)
- `core/runtime` 迭代执行器并发/批内同步
- 与 `KVCachePlugin`(Paged)对上批量 append/load
- `core/tuning/HardwareAwareCache` 落地 `get/put`(§6.3)

## 4. 接口要点

- `Scheduler::submit / plan_next_iteration / execute_iteration / on_request_complete / preempt`(§3.2)
- `BatchPlan { prefills, decodes, transfers, token_budget }`——token_budget 决定一次迭代的容量上限。
- `RequestId` 生命周期管理(§3.2)。

## 5. 实现要点

- **决策循环**:每迭代 `plan_next_iteration()` 依据当前请求集 + token_budget 打包;`execute_iteration` 单次前向覆盖整批(Softmax/matmul 按批展开)。
- **逐请求状态**:位置/已完成掩码/key-value 独立;decode 时逐请求前进,完成请求的 token 计为输出。
- **完成即出批**:`on_request_complete` 释放该请求的 KV 页(块→空闲池)。
- **budget 守约**:受 token 上限约束,避免 OOM(结合 §20.8 KV 预算)。
- **确定性**:同一到达顺序/seed,调度与输出确定(批次不变性友好)。

## 6. 测试与验证

- 吞吐基准脚本(多请求、不同长度),对比:连续批 vs 逐串行;跑出 QPS/p99 与提升倍数。
- 正确性:批内每请求输出 == 单独跑该请求的输出(容差/逐 token)。
- 状态机:请求加入/完成/加入边界行为;budget 边界不超分配。
- 长期运行稳定性(长上下文连续批 + 频繁进出批)无泄漏/崩溃。

## 7. 依赖

- P7(PagedKVCache 批量增删、BlockTable 释放)、P6。
- 基准纳入 `benchmarks/transformer/`(§15)。

## 8. 决策点(默认建议 · 待实现时裁定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现 P8 时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| Chunked Prefill | **本里程碑不实现**,连续批为整段 prefill | 长 prompt 吞吐优化留后续 |
| 请求抢占/优先级 | **不做抢占** | 只用"完成即出批";`preempt` 保留默认空实现 |
| 吞吐/时延度量口径 | **Experiment.Result 统一口径**(P7 候选) | latency p50/p95/p99、QPS、平均/最大批大小,对齐 §13.4 |
| 缓存持久化 | **落盘**(P7 候选) | HardwareAwareCache 跨会话复用 |

## 后续衔接
- P9+ 混合(P11 专家预取、P12 KV 分层)在此批处理框架上叠加;P14 做传统+Transformer 混合编排。