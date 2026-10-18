# 横切设计准则

> 所有里程碑共用的设计约束,源自 `docs/readme.md` §0/§1/§17/§18/§19/§21。实现各里程碑时**先读本节**,再读对应 Spec。

## 1. 双模型支持的总策略(§0,§21)

> **统一 IR 和执行器,分离调度策略和内存策略。** 传统模型与 Transformer 共享约 90% 核心(IR/Graph/Backend/Allocator/Tensor/OpRegistry/KernelRegistry/PluginRegistry/HardwareProfile),差异集中在约 10%:

| 维度 | 传统模型 | Transformer | 统一策略 |
|---|---|---|---|
| 图结构 | 静态 DAG,可能含控制流 | 静态 DAG,含动态维度 | IR 支持控制流+动态维度 |
| 算子 | Conv/Pool/BN/LSTM/GRU/GBDT | Attention/RoPE/LayerNorm/FFN | 统一 OpRegistry,按能力位注册 |
| 执行模式 | 单次执行,固定形状 | 迭代执行(prefill/decode) | 执行器支持两种模式 |
| 批处理 | 固定 batch | 动态 batch/连续批 | 调度器支持两种策略 |
| 内存模式 | 逐层处理,内存稳定 | KV Cache 随序列增长 | 内存抽象支持两类 |
| 量化 | INT8 PTQ/QAT | FP8/KV量化/激活量化 | 统一 QuantPlugin |
| 精度 | FP32/INT8 为主 | FP16/BF16/FP8 为主 | 统一 DType 体系 |
| 瓶颈 | 计算密集(Conv) | 内存带宽密集(Attention/FFN) | 优化策略按瓶颈分档 |

`ModelKind` 枚举:`TRADITIONAL / TRANSFORMER / HYBRID / RULE / UNKNOWN`。**只用于选择优化/内存/调度策略,不进入 IR 执行决策。** Node 上的 `kind` 字段仅作标注,供 HybridPlanner 等按子图分类。

## 2. 能力驱动决策(§2,§18)

- `HardwareProfile` 是唯一决策输入:身份字段(名称/代际)**只写日志,不参与决策**;决策只看 `caps`(能力位)+ `matrix/mem/...`(性能参数)。
- 能力位归一化:跨厂商等价语义统一成同一能力位(NVIDIA TMA ↔ AMD bulk ↔ Intel 2D block)。
- `KernelVariant` 用"能力前置条件"(required_caps/forbidden_caps + 阈值)声明适用性,**不绑定架构名**;可用 `ModelKind` 做二次过滤。
- 指纹生成 = 能力位 + 性能参数量级 + 驱动版本,避免架构名误判。

## 3. 推理零开销保证(§21)

推理路径上必须保证:
- Tensor.`requires_grad=false`;Graph.`is_training=false`;Backend.`supports_training=false`
- 推理**不创建** AutogradEngine;推理**不加载**训练/校验/转换插件
- 推理加载只读 `model_weights`;校验仅 `enable_validation=true` 时启用
- 训练内存使用独立 `MemoryKind`(GRADIENT/OPTIMIZER_STATE/ACTIVATION),按需分配

## 4. 特性门控(§12.3)

新能力一律经 `FeatureFlags` 门控(`enable_experimental=false` 为默认),`docs/readme.md` §12.3 有完整字段。实验特性放 `experimental/` 隔离目录。

## 5. 评审与流水线(§0 决策,§19 风险)

- **正确性基准**:CPU 后端是 oracle,逐算子对比;数值一致性用 Comparator。
- **演进策略**:新特性 = 新实现 + 配置切换;不改接口、不改上层。
- **避免过度抽象/过早优化**:naive → profiler → 优化。
- **接口冻结**:P0.5 之前允许调整,之后冻结。
- **插件 ABI**:C ABI + C++ 内部接口,语义化版本 + 兼容性检查。

## 6. 通用接口与类型(§3,§4.1)

- `Status = std::expected<void, Error>`;`Result<T> = std::expected<T, Error>`
- `Device = {CPU, CUDA, VULKAN}`;`DType` 含 `BOOL`(控制流条件)
- `ModelKind` 见 §2.3
- 所有可演进接口(Backend/Op/Scheduler/KVCache/采样器等)默认纯虚 + 合理默认实现,空实现与默认方法必须在 P0.1 前铺满。

## 7. 关键风险与应对(§19)

| 风险 | 应对 |
|---|---|
| 能力位探测失败 | 回退最保守标量实现 |
| 能力位与性能不匹配 | 运行期实测校准,动态调整 |
| 跨厂商归一化错误 | 保守映射,宁缺毋滥 |
| 指纹过细致缓存失效 | 性能参数量级化 |
| Vulkan barrier 复杂 | 初期保守全局 barrier |
| 数值一致性 | CPU oracle 逐算子对比 |
| 混合负载失衡 | 贪心分配+动态调整 |
| 编译时间膨胀 | 后端独立静态库,核心不依赖 |
| 接口冻结过早 | P0.5 前允许调整 |
| 训练污染推理 | 训练字段默认关闭 |
| 反向图构建复杂 | backward 默认 Unsupported |
| 参考追踪开销 | 仅 `enable_validation` 时启用 |

## 8. 顶层铁律速查

1. 优化决策只看能力画像(能力位+性能参数)。
2. 模型执行只看 IR 语义(不依赖模型类别)。
3. GPU 单后端(CUDA 或 Vulkan),CPU 常驻。
4. 接口面向未来,实现面向当下;P0.5 后台冻结。
5. 可选链路(训练/校验/转换)推理零开销。