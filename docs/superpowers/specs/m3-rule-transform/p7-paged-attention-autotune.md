# P7 · PagedAttention + RuntimeAutotuner(KV 显存利用率 + 运行期微调)

> 设计来源:`docs/readme.md` §5.2(Attention 变体 flash/paged)、§6(自适应调优三层)、§14(PagedAttention 中难、内核自动调优低难)、§20.8。

## 1. 目标与验收(DoD)

- **目标**:用 `PagedKVCache` 替换 RingKVCache,提升显存/KV 利用率;实现运行期微调(RuntimeAutotuner)对未覆盖 shape 在线 benchmark 并缓存。
- **验收标准**:
  1. `PagedKVCache`(§3.1 未来实现)实现 `append/load/clear/memory_usage + block_table/evict_blocks/lookup_prefix/register_prefix`;与 RingKVCache 输出**结果等价**。
  2. **显存利用率提升**:对等序列/并发量下,Paged 的 `memory_usage()` 峰值/分配小于 Ring 峰值(构造成略大的张量以制造片段化差异)。
  3. `BlockTable`、`BlockId` 语义正确;按需分配页,不做整段预留。
  4. `RuntimeAutotuner.tune(op,dtype,shape,profile,kind)` 对候选 kernel 运行 mini-benchmark 返回最快者;结果进 `HardwareAwareCache`(§6.3)。
  5. 内核选择本身是**能力位+实测**,不新写架构名分支(§6)。

## 2. 范围与边界

- **范围内**:`PagedKVCache`、BlockTable 管理、PagedAttention 计算入口、RuntimeAutotuner、HardwareAwareCache(内存态,落地 `save/load` 后可持久化)。
- **范围外**:Paged attention 与连续批协同的调度收益(P8);前缀缓存命中率调优的完整语义(§20.12/§6.4 后继);KV 量化(后置)。

## 3. 需实现/落地的模块

- `plugins/kv_cache/PagedKVCache`(或 `ops/kv_cache`)
- `core/memory/BlockTable`,`BlockId`,页分配器(复用 Allocator/MemoryKind::KV_CACHE)
- `ops/transformer/Attention`(paged 变体,能力位入口)
- `core/tuning/RuntimeAutotuner`、`core/tuning/HardwareAwareCache`(§6.3)
- `core/hardware/RuntimeHardwareProfiler`(性能回填,§6.4)

## 4. 接口要点

- `KVCachePlugin` 扩展方法(§3.1):`block_table()/evict_blocks/compress/lookup_prefix/register_prefix/offload/prefetch/hash_state` 在 Paged 上落地主方法,未用方法保默认。
- `RuntimeAutotuner.tune(...)` 与 `HardwareAwareCache.{get,put,save,load}`(§6.3)。

## 5. 实现要点

- **等价性**:Paged 与 Ring 的**逻辑结果**必须逐 token 一致;差异只在存储分配方式。
- **分页**:按 block 分配到非连续页;`BlockTable` 映射逻辑块→物理页;decode 追加只新增页。
- **autotune 预算**:`TunerConfig` 默认 `max_tuning_iterations=3, tuning_budget_ms=100`(§6.3);超预算即停并用当前最优。
- **缓存键**:`{fingerprint, op, dtype, shape, kind}`(§6.3 `Key`),先内存态,持久化走 `save/load`。
- **能力位过滤**:tune 前先按 KernelVariantRegistry 能力前置条件筛选候选(§6.2),绝不把不支持的变体放进候选。

## 6. 测试与验证

- Paged vs Ring:结果等价(多序列多长度)。
- BlockTable 语义:追加/复用/evict 后状态正确、无 UAF/越界。
- Autotuner:候选顺序确定性;返回值缓存命中时直接返回不重测;预算内停止。
- 显存提升断言:构造含片段化场景,峰/分配比 Ring 低。

## 7. 依赖

- P6(迭代执行器、KVCachePlugin 实现、Attention naive)。
- GPU 可用时,autotune 覆盖后端;CPU-only 也可跑(候选为 CPU kernel 变体)。

## 8. 决策点(默认建议 · 待实现时裁定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现 P7 时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| Autotune 预算/采样 | **用默认 TunerConfig** | `max_tuning_iterations=3`、`tuning_budget_ms=100`;超预算停用当前最优 |
| Autotune 缓存持久化 | **落盘到 tuning_cache_dir** | `HardwareAwareCache` 的 `save/load` 跨会话复用;键=指纹+op+dtype+shape+kind |
| RuntimeHardwareProfiler 覆盖 | **性能回填即可** | 用后端实测 tflops/带宽/时延回填;power/temp(NVML/RAPL)后置 |
| 吞吐/时延度量口径 | **Experiment.Result 统一口径** | latency p50/p95/p99、throughput(QPS)、平均/最大批大小,对齐 §13.4 |

### 缓存失效策略(若采纳落盘持久化)
- 指纹变化(硬件画像含驱动/性能量级)→缓存键失效。
- 单条条目生命周期与 `tuning_cache_dir` 文件一致;损坏/版本不符条目跳过并重建。

## 后续衔接
- P8 在 PagedKV + 连续批下衡量吞吐;P12 用 HybridKVCache 分层。