# 模型转换层(ModelConverter + 五阶段验证)

> 设计来源:`docs/readme.md` §11(模型转换层)、§11.3(五阶段)、§11.4(目录)、§12.2(新架构接入)。转换**可选加载**,作用于模型文件层,不进入推理热路径。

## 1. 目标与验收(DoD)

- **目标**:主流格式(ONNX/TFLite/Caffe/SafeTensors/GGUF)与本引擎统一 IR/格式互转;每个转换都经五阶段验证。
- **验收标准**:
  1. `ModelConverter::load_model(path, cfg)` 统一加载入口可用;`convert_onnx_to_ie` / `convert_hf_to_gguf`(等)按提供的 API 可用(§11.2)。
  2. **五阶段转换验证**可用:`validate_structure / validate_weights / validate_layer_outputs / validate_end_to_end / validate_tokenizer_metadata`(§11.3)。
  3. `ModelLoaderPlugin`/新格式可独立加载(§12.1,衔接 P17)。
  4. 转换后的模型可被推理链路装载执行(输出与源模型参考一致)。

## 2. 格式覆盖(§11.1)

| 格式 | 生态 | 模型类型 | 优先级 |
|---|---|---|---|
| ONNX | ONNX Runtime/Triton/OpenVINO | 传统+Transformer | 高(先) |
| TFLite | TFLite/移动端 | 传统 | 中 |
| Caffe | 经典 CV | 传统 | 低 |
| SafeTensors | HuggingFace | Transformer | 高(先) |
| GGUF | llama.cpp/Ollama | Transformer | 高(先) |
| TensorRT Engine | TRT-LLM | Transformer | 低 |

## 3. 需实现/落地的模块(§11.4)

- `converter/model_converter.h`、`format_loader/{onnx,tflite,caffe,safetensors,gguf}_loader.h`
- `op_mapping/{onnx,tflite,pytorch,gguf}_mapping.h`(算子映射注册表)
- `validation/{conversion_validator, weight_comparator}.h`(含五阶段)
- `quantization/{int8_quant, gguf_quant, awq_quant, gptq_quant}.h`(INT8/GGUF 先,§12.2 量化入口)

## 4. 实现要点

- **算子映射**:解析源图→底层统一算子;未知算子报错而非静默跳过(§20.11 算子映射注册表)。
- **五阶段验证**:结构与权重静态校验 + 层输出/端到端动态数值校验(structure→weights→layer_outputs→end_to_end→tokenizer_metadata,§11.3)。
- **GGUF 量化**:`GGUFQuantType{F32,F16,BF16,Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_0,Q3_K_M,Q2_K}`(§11.2)。
- **目标依赖**:转换后模型交付给推理链路(ONNX 给 P2 传统链路,GGUF/SafeTensors 给 P6 LLM)。转换器本身不承载执行,调用方复用 Backend。

## 5. 加载与校验复用

- 参考追踪 `CrossFrameworkAdapter`(V5)与 `TensorHasher`(V2)为转换验证提供张量/哈希基准(闭环衔接 detection 层)。

## 6. 决策点(默认建议 · 待实现时定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现时在实现内裁定并回写。完整候选清单见 `00-overview/decisions-log.md`。

- 各 loader 算子白名单/映射表规模:候选**以对应推理白名单为界**(ONNX 用 P2 白名单,SafeTensors/GGUF 用 P6 所需);IR 序列化用 JSON(P0 候选)作中间产物。
- `convert_hf_to_gguf` 分词器/元数据迁移:候选**复用自研 BPE 词表迁移(P6)**,metadata 逐字段校对(§11.3/c tokenizer metadata)。