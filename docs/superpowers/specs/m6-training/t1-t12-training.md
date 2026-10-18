# T1–T12 · 训练链路(可选加载,推理零开销)

> 设计来源:`docs/readme.md` §9(训练层)、§16(训练路线 T1-T12 并行)、§21(推理零开销保证)。训练**始终可选**;推理不创建 Autograd、不加载训练插件(§21)。

## 定位与铁律

- 训练与推理**共享内核,分离上层**(§18)。推理算子不自带训练开销;`requires_grad` 默认 false。
- 训练内存走独立 `MemoryKind::{GRADIENT, OPTIMIZER_STATE, ACTIVATION}`(§4.3),按需分配,不复用推理常驻内存。
- `enable_training` / `enable_amp` / FeatureFlags 默认 false。

## 分阶段目标与验收

| 阶段 | 目标 | 验收(DoD) |
|---|---|---|
| T1 | Autograd 基础 | 前向记录 → 反向回传正确(链式法则)于小图;`AutogradEngine::backward(loss)` 得到梯度 |
| T2 | 反向算子覆盖(基础) | Add/Mul/MatMul/Relu/Conv 的反向与数值梯度(中央差分)一致 |
| T3 | 优化器(SGD/Adam/AdamW) | 参数按 `OptimizerConfig` 正确步进;`zero_grad` 清零点 |
| T4 | LR 调度(Cosine/Linear) | `LRSchedulerPlugin::get_lr(step)` 符合曲线 |
| T5 | Loss(CrossEntropy/MSE) | forward/backward 与参考一致 |
| T6 | Dataloader | `next()/shuffle(seed)` 批次正确、种子确定 |
| T7 | Trainer 闭环 | 小模型训练收敛(loss 下降)、`train/evaluate/resume` 可用 |
| T8 | AMP/混合精度 | BF16/FP16 混合精度路径可用且不崩(需后端能力位) |
| T9 | 梯度累积/裁剪 | config 驱动,参数步进与预期一致 |
| T10 | 梯度检查点 | 大图在内存受限下可训,数值一致 |
| T11 | Checkpoint 全量 | 权重/优化器/调度器/RNG/step 保存加载往返一致(`CheckpointIO` §9.4) |
| T12 | 分布式(可选) | DP 起步;TP/PP/ZeRO 后置(§20.9) |

## 关键实现要点

- **反向图构造**:`Graph::build_backward(loss_inputs)`(§3.4)+ `Op::backward`;未实现反向的算子默认 `Unsupported`(§3.6),T2 逐个点亮。
- **AutogradEngine**:`record(op, inputs, outputs)` 记录图;`backward` 拓扑序反向(§9.2)。
- **Trainer**:`Config{epochs, batch_size, lr, optimizer, scheduler, loss, use_amp, grad_accum_steps, checkpoint_interval, checkpoint_dir}`(§9.3)。
- **训练态量化**(LoRA/QLoRA/ZeRO/FSDP,§12.2)划入"前沿可下放",非 T1–T12 必须。

## 决策点(默认建议 · 待实现时定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现对应阶段在实现内裁定并回写。完整候选清单见 `00-overview/decisions-log.md`。

- Autograd 实现方式:候选 **图级反向链**(`record(op,inputs,outputs)`→拓扑逆序回传,对齐 `Graph::build_backward`)。
- 首批反向算子:候选**推理子集对应**(Add/Mul/MatMul/Relu/Conv/Softmax/CrossEntropy/MSE/逐元素)。
- 优化器状态与梯度内存:候选**独立 `MemoryKind::OPTIMIZER_STATE` + 复用梯度内存**;T3/T9 实现时定。
- Checkpoint 格式:候选**二进制权重 + JSON 元数据**(step/epoch/config/RNG);与转换层协同。
- 分布式范围:候选**本期不含分布式**;DP 后续、ZeRO/TP/PP 留前沿。

## 测试与验证

- 梯度正确性:数值梯度(中央差分)对照。
- 训练闭环收敛性测试(固定种子可复现)。
- `keep_inference_zero_overhead`:默认配置下推理路径不实例化训练对象、`requires_grad==false`。