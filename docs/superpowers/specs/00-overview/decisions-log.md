# 决策建议台账(Decision Log)

> **约定(2026-10-18 修订)**:本引擎的工程决策**不在 Spec 期硬锁定**,改为「Spec 只记录**决策点 + 默认建议 + 副作用**,各里程碑**实现时**在其实现内裁定,并回写对应 Spec」。下表是已识别的决策点集合与当前**默认建议(仅供参考,可改)**,不再是硬约束。
> 状态约定:⏳ = 待实现时裁定(默认建议)。表中的"来源"指第一次识别到该点的 Spec。

---

## P0-MVP · 工程选型(来源 `p0-mvp.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| C++ 标准 | C++23 | `std::expected/std::span/std::optional` 可用 | ⏳ |
| 构建系统 | CMake + Ninja | 后端拆独立静态库 | ⏳ |
| 依赖管理 | vcpkg | 可降级为手动子模块 | ⏳ |
| 测试框架 | GoogleTest | CI 友好 | ⏳ |
| 基准框架 | Google Benchmark | P1 起性能 sanity | ⏳ |
| 是否引入 MLIR | 引入 MLIR(自定义 Dialect 多层 IR) | 影响大:构建/依赖/动态 shape 对齐 | ⏳ |
| IR 序列化 | JSON | 与 GraphExporter 对齐 | ⏳ |

## P0.5 · 接口冻结(来源 `p0.5-interface-freeze.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| `Graph::bind` 动态绑定 | 对齐 MLIR 动态 shape | 与 MLIR 决策联动 | ⏳ |
| 插件 ABI 边界 | 提前锁定 C ABI + 语义版本(§12.4) | 降低 P17 返工 | ⏳ |

## P1 · CPU/硬件探测(来源 `p1-cpu-hardware.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| CPU 能力位探测深度 | x86 全量 + ARM 基础(NEON;SVE/SME 接口占位) | AMX 走 CPUID leaf7 | ⏳ |
| Cost model 首版 | 保守启发式(能力位优先+阈值);实测微调留 P7 | | ⏳ |

## P2 · 传统模型链路(来源 `p2-traditional-models.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| ONNX 算子首批白名单 | CNN + 扩展(Upsample/Pad/Clip/Gather/Reduce/Sigmoid/Tanh) | ResNet50 需核心子集 | ⏳ |
| Conv 算法与层融合接入顺序 | im2col+GEMM + Conv+BN+ReLU 融合先行;Winograd/FFT 后置 | | ⏳ |
| 权重常驻内存 | 细化预取(weight_cache_size + enable_weight_prefetch) | 非一刀切全量常驻 | ⏳ |

## P3 · GPU 后端(来源 `p3-gpu-backend.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| GPU 后端选型 | CUDA 且 Vulkan 都实现;运行期激活其一 | 破"二选一"构建约束,保持"单活跃后端"运行约束;工程量↑ | ⏳ |
| §5.3 高级组合路径批次 | 先通用,高级(FP8/TMA/FP4)延后 | | ⏳ |

## P4 · 调度/放置(来源 `p4-scheduler-planner.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| 放置算法 | 保守规则(能力位+简单 cost);贪心/LP/RL 后置 P15 | | ⏳ |
| Cost model | 保守启发式 | | ⏳ |
| 设备选择语义 | 配置优先 + 自动回退 CPU | | ⏳ |

## P5 · 规则引擎(来源 `p5-rule-engine.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| Rete 实现深度 | 先线性(半 left-join),完整 rete 后置 | | ⏳ |

## P6 · LLM 推理(来源 `p6-llm-inference.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| 首批模型架构 | 单一架构 + 一个小权重 | | ⏳ |
| Tokenizer | 自研 BPE(兼容 sentencepiece 词表) | | ⏳ |
| Sampler 策略集 | greedy + top-k/top-p + 温度 | 严格确定性(同 seed) | ⏳ |
| KV 位置/状态 | 绝对位置语义;append 自增;load 只读已append | | ⏳ |

## P7 · PagedAttention + autotune(来源 `p7-paged-attention-autotune.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| Autotune 预算/采样 | 默认 TunerConfig(3 迭代/100ms) | | ⏳ |
| 缓存持久化 | 落盘 tuning_cache_dir | | ⏳ |
| Profiler 覆盖 | 性能回填即可;power/temp(NVML/RAPL)后置 | | ⏳ |
| 度量口径 | Experiment.Result 统一口径 | | ⏳ |

## P8 · 连续批(来源 `p8-continuous-batch.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| Chunked Prefill | 本里程碑不实现,整段 prefill | | ⏳ |
| 抢占/优先级 | 不做抢占(完成即出批);preempt 留默认 | | ⏳ |
| 吞吐口径 | Experiment.Result 统一口径 | | ⏳ |

## P9-P14 · 混合推理(来源 `p9-p14-hybrid-orchestration.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| P11 专家负载均衡 | 贪心 + 热度分区 | 支持 GPU 利用率>60% 验收 | ⏳ |
| P11 残差预取 | 开启,prefetch_depth=1 | | ⏳ |
| 神经元分割 | 本区不实现(长期后置) | | ⏳ |
| P13 投机草稿 | 同架构小权重草稿 | 验收≥1.5x 加速 | ⏳ |
| P14 INT8 PTQ | min/max + Comparator 回退 FP32;AWQ/GPTQ 后置 | | ⏳ |
| P10 跨设备同步 | CUDA stream/event;Vulkan 保守全局 barrier | | ⏳ |

## P15-P18 · 发布/对比(来源 `p15-p18-release.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| 实验存储与统计 | JSON 文件 + 自研轻量统计(均值/方差/置信区间) | | ⏳ |
| CI 基准 | CI 冒烟 + 性能门禁 | | ⏳ |
| 可视化顺序 | 复用 Perfetto/Grafana + 轻量 HTML;自研后置 | | ⏳ |
| 插件动态加载 | dlopen/LoadLibrary + 语义版本 | ABI 建议提前锁定 | ⏳ |
| 发布范围 | C API + ie-* CLI 最小集 | | ⏳ |

## T1-T12 · 训练(来源 `t1-t12-training.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| Autograd 实现方式 | 图级反向链(record→拓扑逆序回传) | 与 Graph::build_backward 对齐 | ⏳ |
| 首批反向算子 | 推理子集对应(Add/Mul/MatMul/Relu/Conv/Softmax/CrossEntropy/MSE) | | ⏳ |
| 优化器状态/梯度内存 | 独立 OPTIMIZER_STATE + 复用梯度内存 | 实现 T3/T9 时定 | ⏳ |
| Checkpoint 格式 | 二进制权重 + JSON 元数据 | | ⏳ |
| 分布式范围 | 本期不含分布式;DP 后续、ZeRO/TP/PP 留前沿 | | ⏳ |

## V1-V7 · 校验(来源 `v1-v7-validation.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| TraceIR 格式 | JSON(复用 IR JSON)meta+ops;张量带数据 | | ⏳ |
| 追踪脚本 | 优先 ONNX 追踪,PyTorch 可选 | | ⏳ |
| 容差默认 | atol=1e-5 / rtol=1e-3,可配(§10.5) | | ⏳ |
| 校验 HTML 报告 | 轻量自研 HTML(复用 P16 方向) | | ⏳ |
| 批次不变性 kernel | 先 Transformer 采样/Attention 核心 | | ⏳ |
| 外部框架输出解析 | vLLM/TensorRT-LLM 的 JSONL 起步 | | ⏳ |

## m8-converter · 转换(来源 `model-converter.md`)
| 决策点 | 默认建议 | 副作用/备注 | 状态 |
|---|---|---|---|
| loader 算子白名单/映射规模 | 以对应推理白名单为界(ONNX 用 P2;SafeTensors/GGUF 用 P6) | | ⏳ |
| convert_hf_to_gguf 分词器/元数据 | 复用自研 BPE 词表迁移(P6);metadata 逐字段校对 | | ⏳ |

---

> 备注:本台账可被实现里程碑回写/覆盖;若实现时决策与默认建议不同,在对应 Spec 的「决策点」节说明原因即可。