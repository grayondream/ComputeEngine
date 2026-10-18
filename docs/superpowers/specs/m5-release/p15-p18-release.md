# P15–P18 · 优化对比 · 发布 · 前沿接入 · 对比框架(收尾)

> 设计来源:`docs/readme.md` §13(可视化/对比/可复现)、§12(可演进/插件/API)、§16(P15–P18)、§6(自适应)。

## 阶段总览

| 子阶段 | 名称 | 目标 | 验收(DoD) |
|---|---|---|---|
| P15 | 优化 + 跨硬件对比 | 编译优化组合 + 跨硬件基准报告 | 出具基准报告 |
| P16 | 发布 | 文档/CI/C API/Web Dashboard | 全流程可复现 |
| P17 | 前沿接入 | 插件生态 + 实验区 | 新插件可独立加载 |
| P18 | 对比框架 | 配置矩阵 + 报告生成 | 一键跑矩阵 |

---

## P15 · 优化组合 + 跨硬件对比

### 目标
把此前的编译优化(常量折叠/算子融合/布局转换等 Pass)组合出可测档位(无/常量折叠/融合/全优化,§13.4),并在多硬件画像上出对比报告。

### 范围
- `PassManager`、`PassRegistry`(§1 编译层)、`PassPlugin`(§12.1)
- Pass:常量折叠、算子融合(Conv+BN+ReLU 等)、布局转换、图重写(§12.2 编译优化)
- 跨硬件基准:`ExperimentRunner::run_cross_hardware` + `ComparisonReport::to_cross_hardware_report`(§13.3)

### 验收
1. 至少无/常量折叠/融合/全优化四档,每档给出算子数/launch 数/延迟(§13.4 编译优化场景)。
2. 跨硬件报告(同一模型在 ≥2 张不同能力硬件画像上的性能/精度)可生成(§13.4 跨硬件对比)。

### 实现要点
- Pass 顺序与正确性:每个 Pass 必须保持 IR 语义(以未优化图 oracle 校验),带等价性测试。
- 编译优化分档 = FeatureFlags/Experiment.config.passes 组合,`.experiment` 记录可复现(§13.5).

---

## P16 · 发布

### 目标
让整套引擎可被他人复现、安装、集成调用。

### 范围
- 文档集(API/构建/使用/架构),`packaging/`(vcpkg/conda/docker 起步)、`api/`(C API 完整 + `cli`)、`benchmarks/` 落地
- `ie-*` CLI(§13.6)可用:`ie-run/ie-report/ie-trace/ie-graph/ie-probe/ie-tune/ie-hash/ie-verify/ie-determinism`(子集按已完成能力)
- C API:§12.4 全量导出 + 版本兼容机制

### 验收
1. 干净环境按文档可从源码构建并跑通 MVP + 关键 CLI。
2. C API 最小示例可编译链接调用 `ie_create_backend`。
3. 实验元数据含 git_commit/build_type/compiler/hardware_profile/input_hash(§13.5),可复现。

### 实现要点
- 后端独立静态库避免核心依赖膨胀(§19 编译时间)。
- 打包以最小面集合起(deb/rpm 可后置)。

---

## P17 · 前沿接入(插件生态 + 实验区)

### 目标
开放可演进架构(§12),让新特性以插件独立接入,不改接口不改上层。

### 范围
- 插件注册表全开:`OpPlugin/BackendPlugin/PassPlugin/QuantPlugin/SchedulerPlugin/KVCachePlugin/SamplerPlugin/ModelLoaderPlugin/TokenizerPlugin/OptimizerPlugin/LRSchedulerPlugin/LossPlugin/DataloaderPlugin/HardwareProbePlugin`(§12.1)
- `PluginRegistry::load_shared_library`(动态加载,§4.4)+ ABI 版本兼容(§12.4)
- `experimental/` 实验区 + `FeatureFlags` 门控(§12.3)

### 验收
1. 一个示例新插件(如 FlashAttention-2 / 或新 Conv)可**独立编译并加载**进引擎而不动核心代码。
2. ABI 版本检查:不兼容插件被拒绝并给出清晰错误。

### 实现要点
- 插件 ABI:相交面用 C ABI + 内部 C++ 接口(§18 插件边界);加载失败回退安全路径。
- 版本兼容:`,IE_API_VERSION_*` 语义化 + 兼容性检查(§19 插件 ABI 风险)。

---

## P18 · 对比框架

### 目标
配置矩阵 + 实验仓库 + 一键报告,规模化对比(§13.3/§13.6)。

### 范围
- `ExperimentStore`(save/load/list/export_csv/export_json/find_by_capability/find_by_bandwidth_range/find_by_model_kind,§13.3)
- `ExperimentRunner::run_matrix / run_ablation / run_cross_hardware`、`ComparisonReport`(markdown/html/csv/compare)
- `ie-report --input ... --format html`(§13.6),结果落 `results/`

### 验收
1. 一条命令跑通配置矩阵(设备×策略×量化×调度×KV)并生成 HTML 报告(§13.6 `ie-run --matrix` + `ie-report`)。
2. 消融实验(run_ablation)与跨硬件对比可用;`find_by_*` 查询可用。
3. 关键对比场景(§13.4)至少覆盖:后端、量化、调度、KV、编译优化、框架对齐、确定性、跨硬件。

### 实现要点
- 多次运行统计(方差/p50/p95/p99,§20.16)纳入;实验 id 含时间戳/随机后缀(§13.5 可复现)。
- 框架对齐(vLLM/TensorRT-LLM/ONNX Runtime)通过 CrossFrameworkAdapter 输出对照(§10.6),起步对齐 ONNX Runtime/PyTorch,再扩(§19 跨框架对齐)。

---

## 贯穿决策点(默认建议 · 待实现时定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现对应子阶段时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| P18 ExperimentStore 存储与统计 | **JSON 文件 + 自研轻量统计** | 结果存文件;统计含均值/方差/置信区间;不用 sqlite |
| CI 自动基准 | **CI 冒烟 + 性能门禁** | CI 只跑关键场景冒烟 + 性能上限防回归;全量矩阵手动 |
| P16 可视化顺序 | **复用 Perfetto/Grafana + 轻量 HTML** | 起步用 Perfetto/Grafana + 轻量 HTML 报告;自研 Web Dashboard 后置 |
| P17 插件动态加载 | **dlopen/LoadLibrary + 语义化版本** | ABI 建议提前锁定(P0.5);load 不兼容插件报错回退 |
| P16 发布范围 | **C API + ie-* CLI 最小集** | 首版发布 C API、`ie-*` CLI、CMake 安装/头文件;sdk/容器/wasm 按需后置 |