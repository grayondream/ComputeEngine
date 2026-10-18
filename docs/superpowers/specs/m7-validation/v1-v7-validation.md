# V1–V7 · 校验与调试链路(可选加载)

> 设计来源:`docs/readme.md` §10(校验层)、§16(校验路线 V1-V7)、§11.3(五阶段转换验证)。校验**仅在 `enable_validation=true`** 时启用;参考追踪开销仅在该开关打开(§19)。

## 定位与铁律

- 四层校验管线(§10.1):任务级 → Prompt 级 → 张量级 → 权重级。
- 参考追踪解耦框架(`TraceIR` 不依赖 PyTorch/ONNX,§10.2);`CrossFrameworkAdapter` 对齐外部框架。
- 确定性模式默认关;调试时开(§19 性能权衡)。

## 分阶段目标与验收

| 阶段 | 目标 | 验收(DoD) |
|---|---|---|
| V1 | TraceIR + ReferenceTracer 骨架 | `TraceIR` 结构完整;`trace_onnx` 采样小模型出 trace;`save/load` 往返 |
| V2 | Tensor 哈希 + 权重验证 | `TensorHasher`(XXH64 默认/SHA256)对模型埋入 `HashManifest`;`verify_manifest` 可检出被篡改张量 |
| V3 | 张量级对比 Comparator | `compare_tensor` + `find_first_divergence`;容差(rtol/atol)可配;`CompareReport` 生成 |
| V4 | 确定性(DeterminismGuard) | `enable_deterministic` 后同 seed 输出逐 token 确定;`verify_batch_invariance` 可用 |
| V5 | 跨框架对齐(起步 ONNX/PyTorch) | `from_onnxruntime/from_huggingface` 适配;首分歧点报表(§13.4 框架对齐) |
| V6 | HTML 校验报告 | `validation/report` 生成逐层误差曲线/张量分歧热力图(§13.2 校验行) |
| V7 | 四层机成熟 | 任务级 LLM 评测 + Prompt 级 → 张量级 → 权重级一键管线;`ie-verify` 可用 |

## 关键实现要点

- **确定性保证**(§10.4):`DeterminismConfig{ deterministic_kernels, batch_invariant, disable_tf32, use_ieee_rounding, disable_prefix_caching }`;批不变性 Kernel(§20.10)在 `supports_batch_invariant` 后端可用。
- **Trace 捕获**:仅 `enable_validation` 与 `capture_kv_cache` 时采集 KV(§10.7),避免常态开销。
- **容差校准**(§20.10):默认 `atol=1e-5, rtol=1e-3`,`Comparator.CompareConfig` 可调。
- **跨框架适配**(§10.6):`from_vllm / from_tensorrt_llm / from_huggingface / from_onnxruntime` → `TraceIR`;起步 ONNX Runtime + HF,再扩 vLLM/TensorRT-LLM(§19 跨框架对齐).

## 决策点(默认建议 · 待实现时定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现对应阶段在实现内裁定并回写。完整候选清单见 `00-overview/decisions-log.md`。

- TraceIR 格式:候选 **JSON**(复用 IR JSON)meta+ops,张量带数据(V1 定)。
- 追踪脚本:候选**优先 ONNX 追踪,PyTorch 可选**。
- 容差默认:候选 `atol=1e-5 / rtol=1e-3`,可配(§10.5)。
- 校验 HTML 报告:候选**轻量自研 HTML**(与 P16 方向一致)。
- 批次不变性 kernel:候选**先 Transformer 采样/Attention 核心**。
- 外部框架输出解析:候选 **vLLM/TensorRT-LLM 的 JSONL 起步**(V5 定)。
- 任务级评测指标:按 `metrics/` 模块落地(§15 observability/metrics)。

## 测试与验证

- trace 往返一致性(load==save);哈希篡改检出;对比器首分歧点正确;确定性子集逐 token 复现。
- 预算:校验链路在 `enable_validation=false` 时零开销注入(编译/运行均可证)。