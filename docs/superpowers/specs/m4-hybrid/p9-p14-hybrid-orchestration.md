# P9–P14 · 混合推理(两设备协同)与混合编排

> 设计来源:`docs/readme.md` §7(混合推理层)、§16(P9–P14)、§14(借鉴优化项)、§12.2(并行/异构接入)。本 Spec 按阶段切分并串成一条线。**两设备 = CPU(常驻) + 单一 GPU(§1 设备拓扑)。**

## 阶段总览

| 子阶段 | 名称 | 目标 | 验收(DoD) |
|---|---|---|---|
| P9 | 混合 H1:层间分割 | HybridPlanner 按层放置 | CPU+GPU **串行**正确 |
| P10 | 混合 H2:异步 | Event/流同步 | 设备**并行** |
| P11 | 混合 H3:专家分割 | 专家分置+残差预取 | GPU 利用率>60% |
| P12 | 混合 H4:KV 分层 | HybridKVCache | 长上下文不 OOM |
| P13 | 混合 H5:投机 | SpeculativeDecoder | 加速≥1.5x |
| P14 | 混合编排+INT8 PTQ | HybridOrchestrator | 传统+Transformer 混合链路 |

---

## P9 · H1 层间分割(pipeline 雏形)

### 目标
`HybridPlanner` 在 CPU/GPU 间做**层间**分割(§7.1 层间分割:串行,利用率低,但正确性第一)。

### 范围
- `HybridPlanner::plan()`(§7.3):`candidate_devices/analyze_op_types/classify_model_kind/decide_placement/balance_load(最小)/insert_transfers`
- `HybridConfig`(§7.2)落读:STRATEGY、DeviceConfig(memory_budget/compute_share)、manual_placement
- `HybridScheduler`(§7.4)串行执行:CPU executor → transfer → GPU executor → transfer …

### 验收
1. 同一模型按 manual_placement 分置两设备,端到端结果与单设备一致。
2. `NodePlacement{node, backend, estimated_cost, memory_footprint, needs_transfer}` 结构正确(§7.3)。
3. 传输插入正确(跨设备依赖边仅一次 copy)。

### 实现要点
- 层间切分用 LayerPlacement(§7.2)做边界;先不追求负载均衡。
- TransferManager(`transfer`)完成跨设备搬运(§7.5 矩阵)。

---

## P10 · H2 异步(Event/流同步,设备并行)

### 目标
让两设备**并行**执行各自片段;用 Event 做跨设备依赖同步(§18 Compute-transport overlap,§14 计算传输重叠)。

### 范围
- 后端 async:`execute_async` + `wait(Timeout)` + 多流/queue(§3.3)
- `EventBus`、`DeviceQueue`(cpu_queue/gpu_queue,§7.4)与 Event 原语

### 验收
1. 两设备时间线重叠(火焰图/时间线显示并行区间),而非 P9 串行。
2. 用 Chrome Trace(§13)验证重叠片段。

### 实现要点
- 传输与计算重叠(HiCache 参考,§14):发送下下个块的同时算当前块。
- 保守 barrier 起步(Vulkan 全局 barrier),后续再精化(§19)。

---

## P11 · H3 专家分割(MoE)

### 目标
MoE 场景把专家分置两设备,真正并行(**专家分割是首选并行**,§7.1);残差预取 + 负载均衡。

### 范围
- `MoE_Router/MoE_Expert` 算子执行(§4.2 OpType)
- `ExpertConfig{max_experts_on_gpu, hotness_threshold, enable_prefetch}`(§7.2)
- `balance_load`(贪心)与残差预取

### 验收
1. GPU 利用率 >60%(基准报告,§13.4 混合策略对比)。
2. MoE 前向正确(专家结果合并后与单设备一致)。
3. 残差预取不破坏语义。

### 实现要点
- 专家按热度(训练统计或运行期采样)分置;负债均衡贪心(§19 风险).
- MoE 数值合并(Softmax over router logits × expert out)与单设备一致。

---

## P12 · H4 KV 分层(HybridKVCache)

### 目标
KV 在 GPU/CPU 间分层放置,长上下文不 OOM(§7.2 kv_cfg)。

### 范围
- `HybridKVCache`(§3.1 KVCachePlugin):`append/load/evict_blocks/offload/prefetch` 跨设备
- `KVCacheConfig{gpu_kv_budget, offload_to_cpu, cpu_kv_budget}`(§7.2)
- 卸载/预取:`offload(layer,start,end)`、`prefetch(layer,start,end)`(§3.1)

### 验收
1. 超 GPU 预算的长上下文:持续生成不 OOM(层/块卸载到 CPU)。
2. 卸载后 load 结果与不卸载等价(数值一致)。

### 实现要点
- 块粒度决策:冷块卸载 CPU,热块预取回 GPU(结合 §12 RTC 前缀缓存)。
- 命中率/显存指标纳入 Experiment 口径(§13.4 KV Cache 对比)。

---

## P13 · H5 投机解码(SpeculativeDecoder)

### 目标
异构各司其职(§7.1 投机解码):草稿模型快速探索 + 目标模型验证,加速 ≥1.5x(§16)。

### 范围
- `SpeculativeDecoder` + `SamplerPlugin.sample`(投机入口,§12.1)
- `SpeculativeConfig` / FeatureFlags `enable_speculative`

### 验收
1. 端到端加速 ≥1.5x(同 batch/宽度),且**拒绝率受控**下输出分布质量不显著退化。

### 实现要点
- 草稿/目标按延仿真一致采样;接受率与加速、成本模型(TODO,§20.6)。
- 严格正确性:接受的新 token 与"一次一 token"策略在自回归语义上一致(接受时才提交)。

---

## P14 · 混合编排 + INT8 PTQ

### 目标
`HybridOrchestrator` 把规则/传统/Transformer 编排成一条统一图链路(§8.4);INT8 PTQ 量化注入传统子图(§12.2/§11.4)。

### 范围
- `HybridOrchestrator::build_pipeline(PipelineSpec)`、`run(Graph, Input)`(§8.4,统一图表示编排)
- 编排算子:`ToolCall, LLMGenerate, Route, Merge`(§4.2 OpType)
- `QuantPlugin`(INT8 PTQ 校准:`int8_quant.h`,§11.4)针对**传统子图**

### 验收
1. 一条"传统模型特征 → 规则路由 → LLM 生成/工具调用"的混合链路可执行并在统一图里可追踪。
2. INT8 PTQ 后同子图数值在容差内、显存/延迟获益(基准)。
3. 混合链路可观测(`set_model_kind` 标注,§13.1)。

### 实现要点
- INT8 PTQ 校准策略(§20.7)起步用全层贪心(min/max 或百分位),精度回退校验由 Comparator 兜底。
- 编排图沿用统一 IR;`Route/Merge` 负责范式边界数据流转。

---

## 贯穿决策点(默认建议 · 待实现时定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现对应子阶段时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| P11 专家负载均衡 | **贪心 + 热度分区** | 专家按运行期热度贪心分置(热→GPU,冷→CPU),支持"专家分割+残差预取";GPU 利用率>60% 验收以此支持 |
| P11 残差预取 | **开启,prefetch_depth=1** | 下一批专家残差逐块预取与计算重叠(借鉴 KTransformers/DALI) |
| 神经元分割 | **本区不实现**(长期后置) | 需 profiling,不做 dense LLM 细颗粒拆分 |
| P13 投机草稿 | **同架构小权重草稿** | 草稿=同架构小权重,目标=同架构大权重;加速按同 token 验收 |
| P14 INT8 PTQ | **min/max(或百分位)+ 回退** | 全层贪心校准;经 Comparator 容差验证(atol1e-5/rtol1e-3),失败回退 FP32;AWQ/GPTQ 后置 |
| P10 跨设备同步 | **保守同步起步** | CUDA 用 stream/event;Vulkan 初期保守全局 barrier,后续精化依赖级 |

> 注:切分策略的放置/成本沿用 P4 建议(保守规则+保守 cost);`§20.6` 进阶留优化里程碑。