# P2 · ONNX 加载器 + 传统模型算子(Conv/Pool/BN)

> 设计来源:`docs/readme.md` §4(OpType 传统算子)、§8.2(TraditionalEngine)、§11(转换/加载)、§14(借鉴传统模型优化)、§20.7(传统模型 TODO)。

## 1. 目标与验收(DoD)

- **目标**:打通"ONNX 模型 → 加载为 IR → CPU 执行"的传统模型推理链路;Conv/Pool/BN 等算子正确;ResNet50 对齐 ONNX Runtime。
- **验收标准**:
  1. **ResNet50 对齐 ONNX Runtime**:给定输入,本引擎 logits 与 ONNX Runtime 输出在容差内一致;端到端分类 Top-1 一致。
  2. ONNX 加载器能将标准 Conv/Pool/BN/Add/Relu 子图转为 IR 并执行。
  3. `TraditionalEngine::load_model/infer/infer_batch` 可用。
  4. **算子对齐测试**:Conv2D / MaxPool / AvgPool / BatchNorm / GlobalAvgPool 逐层与 ONNX Runtime 的 layer output 对齐。

## 2. 范围与边界

- **范围内**:ONNX loader(先覆盖传统模型子集)、Conv/Pool/BN/Add/Relu/Softmax/Concat/Slice/Reshape/Gemm 算子、`TraditionalEngine`、`MemoryKind::WEIGHT_CONST` 权重常驻、批量推理。
- **范围外**:TFLite/Caffe 加载器(目录就位);Transformer 相关(§11 表注明**传统模型走 ONNX/TFLite/CAFFE**);LSTM/GRU 真实执行(P 阶段控制流接入,见 §20.3/§20.7)。
- **基准**:正确优先;`naive → profiler → 优化`。INT8 量化留 P14/P1;`enable_int8_ptq=true` 但实现门控。

## 3. 需实现/落地的模块

- `models/traditional/`:ONNX loader(优先;TFLite/Caffe loader 留接口,§11.4)
- `ops/traditional/`:`Conv2D, DepthwiseConv2D, ConvTranspose2D, MaxPool2D, AvgPool2D, GlobalAvgPool, BatchNorm, GroupNorm（LSTM/GRU 后置）`
- `ops/common/`:`Concat, Slice, Reshape, Transpose, Gather, Reduce, Sigmoid, Tanh, Clip, Pad, Upsample`(按 ONNX 子集需要)
- `ops/traditional/static`:`TraditionalEngine`(推理范式层,§8.2)
- `converter/op_mapping/onnx_mapping.h`(算子映射注册表起步,§11.4)

## 4. 接口要点

- `ModelFormat::{ONNX,...}` 与 `TraditionalEngine::load_model(path, format)`、`infer(inputs)`、`infer_batch(batch)`(§8.2)
- `ModelConverter::load_model(path, cfg)` + `convert_onnx_to_ie`(§11.2)
- `ConversionValidator::validate_structure / validate_weights / validate_layer_outputs / validate_end_to_end / validate_tokenizer_metadata`(§11.3,先在 ONNX 子集上实现)

## 5. 实现要点

- **算子覆盖子集**:先定义 ONNX 算子覆盖清单并逐条对齐(§20.7),首批 = CNN 推理所需;每算子带 layer-level 对照测试。
- **Conv 算法选择**(§14/§20.7):起步 `im2col + GEMM`(🟢 低),`Winograd`/`FFT` 后置;能力位/形状决定算法选择入口留好。
- **层融合**:Conv+BN+ReLU 融合(🟢 低)作为首批 Pass;归一化与融合语义要严格。
- **内存**:权重走 `MemoryKind::WEIGHT_CONST`(常驻)+ 逐层激活从激活池分配(§4.3 `MemoryPolicy` 权重缓存字段)。
- **正确性 oracle**:CPU 传统算子实现即为 oracle;层输出对齐 ONNX Runtime。

## 6. 测试与验证

- 每算子:构造小网络,本引擎 layer output vs ONNX Runtime 逐层比对(容差 `atol/rtol`)。
- end-to-end:ResNet50 输入对齐 + Top-1。
- 批量推理 `infer_batch` 结果与逐样本一致(批次不变性 sanity)。
- 模型加载器对非法/不支持算子的报错路径测试。

## 7. 依赖

- P1(CPU 后端能力位 kernel、MatMul)。
- ONNX Runtime 或等价(作为参考对照;若不可用,退化为 PyTorch/numpy 参考官方权重前向)。
- 需要一个已知 ResNet50 权重(公开模型,CI 只跑 layer 子集以控制体积)。

## 8. 决策点(默认建议 · 待实现时裁定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现 P2 时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| IR 序列化格式 | JSON(P0 候选) | ONNX→IR 中间产物用 JSON |
| ONNX 算子首批白名单 | CNN + 扩展 | 核心集:Conv/BN/MaxPool/AvgPool/GlobalAvgPool/Relu/Add/Mul/Gemm/MatMul/Flatten/Reshape/Softmax/Concat/Slice/Transpose另加 Upsample/Pad/Clip/Gather/Reduce/Sigmoid/Tanh |
| Conv 算法与层融合 | im2col+GEMM + Conv+BN+ReLU 融合先行;Winograd/FFT 后置 | 首批 kernel=im2col;融合作为首批 Pass |
| 权重常驻内存 | 细化预取 | `MemoryPolicy` 落地 `weight_cache_size` 上限 + `enable_weight_prefetch`(逐层预取);非一刀切全量常驻 |

### 白名单算子清单(建议首批)
- 卷积/池化/归并:`Conv2D, DepthwiseConv2D, ConvTranspose2D, MaxPool2D, AvgPool2D, GlobalAvgPool`
- 归一化/激活:`BatchNorm(GroupNorm 可选), Relu, Sigmoid, Tanh, Clip`
- 数据布局:`Flatten, Reshape, Transpose, Concat, Slice, Gather, Pad, Upsample, Reduce`
- 矩阵:`Gemm, MatMul, Add, Mul`