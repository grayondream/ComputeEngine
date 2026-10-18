# P6 · Transformer 算子 + LLM 推理(prefill/decode + RingKVCache)

> 设计来源:`docs/readme.md` §3.1(KVCachePlugin)、§4.2(Transformer 算子)、§8.3(LLMEngine/Sampler/Tokenizer)、§14(借鉴)。

## 1. 目标与验收(DoD)

- **目标**:打通"GGUF/SafeTensors 小模型 → 解码为 IR → 迭代执行(prefill/decode)→ 流式生成";`RingKVCache` 管 KV。
- **验收标准**:
  1. 一个小因果 LLM(如某公开小权重 GGUF)可 `LLMEngine.generate(prompt, params, on_token)` **流式**产出 tokens(§8.3)。
  2. `Sampler` 按 `SamplingParams` 采样(greedy + top-k/p),结果可复现(同 seed)。
  3. `RingKVCache` 实现 `append/load/clear/memory_usage`(§3.1),decode 时正确复用已缓存 KV。
  4. Transformer 算子对 PyTorch 参考对齐:Attention / RoPE / RMSNorm / Embedding(LayerNorm 已 P2)。
  5. **迭代调度器**:`ContinuousBatchScheduler`(或退化的单请求迭代)以 prefill+decode 循环驱动执行(§3.2)。

## 2. 范围与边界

- **范围内**:Transformer 算子、LLMEngine、Tokenizer(BPE 起步)、Sampler、RingKVCache、迭代执行器。
- **范围外**:PagedAttention/PagedKVCache(P7)、连续分批/并发请求(P8)、KV 量化(后置)、投机解码(P13)、FlashAttention 深度(可后置,naive attention 先正确)、MoE(P11+)。

## 3. 需实现/落地的模块

- `ops/transformer/`:`Attention, MultiHeadAttention, RoPE, RMSNorm, Embedding, PositionalEncoding(FFN 用现有 MatMul/Act)`
- `backends/cpu/`(或 GPU)+ `ops/common/MatMul/Softmax` 升级应对长序列带宽
- `llm/`:`tokenizer/(BPE)`,`model_loader/(GGUF 起步,SafeTensors 后置)`,`transformer/(GPT 层)`,`sampler/`,`streaming/`
- `core/scheduler/ContinuousBatchScheduler`(最小可用)、`core/runtime` 迭代执行器
- `core/memory/`:**RingKVCache** + `MemoryKind::KV_CACHE`

## 4. 接口要点

- `KVCachePlugin`(§3.1):本阶段实现 `RingKVCache`;未来 Paged/Quantized/Hybrid 复用同接口。
- `Tokenizer`/`Sampler`/`LLMEngine`(§8.3).
- `Scheduler`(§3.2)双模式:迭代调度器首次真实使用 `prefills + decodes`。
- `OpType` Transformer 算子族已在 P0.5 冻结。

## 5. 实现要点

- **attention naive 先行**:先追求正确(与 PyTorch 逐张量对齐),带宽优化留 P7。
- **KV 复用必须正确**:load(position) 只允许读已 append 范围;clear 语义清晰。
- **数值确定性**:同 seed 同输入,输出 token 序列逐 bit 可复现(涉及采样 RNG 独立流)。需求见 §4.5 SamplingRNG。
- **迭代执行器**:prefill 全序列一次;decode 每步一次前向 + 采样,维护 positions/KV 状态。
- **流式回调**:每 decode 一步触发 `on_token`(可逐 token 输出)。

## 6. 测试与验证

- Attention/RoPE/RMSNorm/Embedding 对 PyTorch 参考张量对齐(容差)。
- KV 语义:append/load 越界与正常路径;decode 结果 == 不共享 KV 的 refork 结果(缓存正确性)。
- 采样:greedy/top-k deterministic;seed 复现。
- end-to-end:小模型流式生成,首 token 可复现,输出前缀合理。

## 7. 依赖

- P2(MatMul/Softmax/逐元素在长序列上的正确性)、P4(迭代执行路径)。
- 需要一个公开小权重 GGUF 及对应 tokenizer(体积可控,用于 CI)。
- 若走 LLM 对齐,需要 PyTorch 参考实现(实验/CI 可选)。

## 8. 决策点(默认建议 · 待实现时裁定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现 P6 时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| 首批模型架构 | **单一架构 + 一个小权重** | 首选一个因果 LLM 架构对接一个小权重 GGUF(体积可控进 CI);同一架构可多权重 |
| Tokenizer | **自研 BPE** | 可复现、无重依赖;兼容读取 sentencepiece 词表与自格式 |
| Sampler 策略集 | **greedy + top-k + top-p + 温度** | top-k/top-p/温度严格确定性(同 seed);额外策略后置 |
| KV 位置/状态语义 | **绝对位置语义** | position= 绝对全局位置;append 自增;load 仅允许读已 append 范围 |

## 后续衔接
- P7 用 PagedKVCache 替换 RingKVCache;P8 加连续批;P13 加投机。保持 `KVCachePlugin`/`Scheduler` 接口不变。