# 推理引擎实现 Spec · 总览

> 本目录是把 `docs/readme.md` 设计文档转译为一整套按里程碑分文件夹、面向**实现者**的中文 Spec。
> 每个里程碑一份独立文档,包含:目标与验收(DoD)、端口与边界、接口冻结清单、需创建的模块与文件、实现要点、测试与验证、依赖、待决 `TODO:`。
> 设计权威仍是 `docs/readme.md`;本 Spec 补充可执行的阶段切分、验收与实现次序,不改变设计语义。

## 目录索引

| 目录 | 覆盖阶段 | Spec 文档 | 一句话目标 |
|---|---|---|---|
| `00-overview/` | 全周期 | `README.md`(本文件)、`cross-cutting.md` | 里程碑地图、验收矩阵、横切设计准则 |
| `m1-bootstrap/` | P0-MVP, P0.5 | `p0-mvp.md`、`p0.5-interface-freeze.md` | 跑通静态图 CPU 推理;冻结全部接口 |
| `m2-backend-core/` | P1, P2, P3, P4 | `p1-cpu-hardware.md`、`p2-traditional-models.md`、`p3-gpu-backend.md`、`p4-scheduler-planner.md` | CPU 后端完善;传统链路;GPU 后端;能力驱动调度 |
| `m3-rule-transform/` | P5, P6, P7, P8 | `p5-rule-engine.md`、`p6-llm-inference.md`、`p7-paged-attention-autotune.md`、`p8-continuous-batch.md` | 规则引擎;Transformer/LLM;PagedAttention+autotune;连续批 |
| `m4-hybrid/` | P9–P14 | `p9-p14-hybrid-orchestration.md` | 两设备混合推理与混合编排 |
| `m5-release/` | P15–P18 | `p15-p18-release.md` | 优化对比、发布、前沿接入、对比框架 |
| `m6-training/` | T1–T12 | `t1-t12-training.md` | 可选加载的训练链路 |
| `m7-validation/` | V1–V7 | `v1-v7-validation.md` | 可选加载的校验/调试链路 |
| `m8-converter/` | 模型转换 | `model-converter.md` | 模型格式转换与验证 |

## 里程碑地图与验收矩阵(源自 `docs/readme.md` §16/§18/§21)

| 里程碑 | 目标 | 最小实现 | 验收标准(DoD) |
|---|---|---|---|
| P0-MVP | 跑通静态图 CPU 推理 | AddOp + CpuBackend + StaticScheduler | `(a+b)*3` 正确 |
| P0.5 | 接口补全+硬件画像+双模型预留 | 空实现+默认方法 | 接口契约测试通过 |
| P1 | CPU 后端完善+硬件探测 | 能力位探测 + AVX2/AVX-512 kernel | 小 MLP 对齐 PyTorch |
| P2 | ONNX 加载器+Conv/Pool/BN | 传统模型推理链路 | ResNet50 对齐 ONNX Runtime |
| P3 | GPU 后端(一选一)+GPU 探测 | 能力位驱动 kernel 选择 | 同图 CPU/GPU 一致 |
| P4 | BackendScheduler+OptimizationPlanner | 能力驱动放置、传输 | CPU+GPU 自动调度 |
| P5 | 规则引擎 | RuleEngine + 推理链可视化 | 输出推理链 |
| P6 | Transformer 算子+LLM 推理 | Attention + RingKVCache | 小模型流式生成 |
| P7 | PagedAttention+RuntimeAutotuner | PagedKVCache 替换 | 显存利用率提升 |
| P8 | 连续批+HardwareAwareCache | ContinuousBatchScheduler | 吞吐提升 |
| P9 | 混合 H1:层间 | HybridPlanner | CPU+GPU 串行正确 |
| P10 | 混合 H2:异步 | Event 同步 | 设备并行 |
| P11 | 混合 H3:专家 | 残差预取+负载均衡 | GPU 利用率>60% |
| P12 | 混合 H4:KV 分层 | HybridKVCache | 长上下文不 OOM |
| P13 | 混合 H5:投机 | SpeculativeDecoder | 加速≥1.5x |
| P14 | 混合编排+INT8 PTQ | HybridOrchestrator+INT8 | 传统+Transformer 混合链路 |
| P15 | 优化+跨硬件对比 | 对比报告/消融实验 | 基准报告 |
| P16 | 发布 | 文档/CI/C API/Web Dashboard | 可复现 |
| P17 | 前沿接入 | 插件生态/实验区 | 新插件可独立加载 |
| P18 | 对比框架 | 配置矩阵/报告生成 | 一键跑矩阵 |

训练 T1–T12、校验 V1–V7、模型转换 与推理路线并行推进,均**可选加载**,不得污染推理路径。

## 架构主线(全周期不变的约束)

```text
核心:IR + Graph + Backend 接口 + 内存抽象 + 调度骨架
硬件:HardwareProfile(能力位+性能参数)→ OptimizationPlanner → 决策
后端:CPU(常驻参考) + 单一 GPU(CUDA 或 Vulkan,二选一)
双模型:统一 IR + 执行器;分离调度策略(静态/迭代)与内存策略(权重常驻/KV Cache)
范式:规则 / 传统 / Transformer / 混合编排,统一图表示
训练/校验/转换:可选加载;推理零开销
演进:插件注册表 + 实验区 + 特性门控
观测:Trace + Graph + Metrics 三类数据流
对比:配置矩阵 + 实验仓库 + 报告(含跨硬件)
```

## 顶层铁律(逐里程碑约束,详见 `cross-cutting.md`)

1. **优化决策只依赖硬件能力画像(能力位 + 性能参数),不依赖架构名/型号/代际名。**
2. **模型执行只依赖 IR 语义,不依赖模型类别**;`ModelKind` 仅影响优化策略选择。
3. **GPU 侧只激活一个后端**(CUDA 或 Vulkan),CPU 常驻。
4. **接口面向未来,实现面向当下**,先跑起来再逐个接入;接口在 P0.5 后冻结。
5. 训练/校验/转换字段默认关闭,推理零开销。
6. 现有代码交付前必须通过验证(见 `verification-before-completion`),并遵循 TDD。

## 决策约定(2026-10-18 定)

- 本引擎的工程决策**不在 Spec 期硬锁定**。Spec 只记录「**决策点 + 默认建议 + 副作用**」。
- 各里程碑**实现时**在其实现内裁定决策,并回写对应 Spec 与 `00-overview/decisions-log.md`(该文件是候选建议台账,非约束)。
- 若要改动任一决策,在对应 Spec 的「决策点」节说明原因即可。
- 各 Spec 末节的「决策点(默认建议 · 待实现时定)」清单即当前候选,实现者从默认建议起步即可。

## 阅读顺序建议(实现者)

总览(`00-overview/`) → 当前批次所属里程碑 Spec(依赖决策建议起步) → 需要时查 `cross-cutting.md` 对应章节 → 回到 `docs/readme.md` 对应原文。