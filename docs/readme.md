# 推理引擎完整设计方案（传统模型 + Transformer 双支持版）

> **定位**：C++ 自研推理引擎，CPU + 单一 GPU（CUDA 或 Vulkan，二选一），支持 CPU+GPU 混合推理，**同时支持传统模型（CNN/RNN/MLP/GBDT）与 Transformer 模型**，统一承载规则推理、张量推理、LLM 推理。预留训练扩展接口，内置调试与一致性校验工具链，推理路径零开销。
> **核心原则**：**优化决策只依赖硬件能力画像（能力位 + 性能参数），不依赖架构名。模型执行只依赖 IR 语义，不依赖模型类别。**
> **策略**：接口面向未来，实现面向当下。先跑起来，再逐个接入新特性。
> **约束**：GPU 侧只激活一个后端，CPU 常驻。
> **说明**：具体技术选型以 `TODO:` 标注。


## 0. 传统模型与 Transformer 的差异分析

同时支持两类模型，需要先明确差异，再决定哪些抽象要统一、哪些要分离。

| 维度 | 传统模型 | Transformer 模型 | 统一策略 |
|---|---|---|---|
| **图结构** | 静态 DAG，可能含控制流 | 静态 DAG，含动态维度 | IR 支持控制流 + 动态维度 |
| **算子类型** | Conv/Pool/BN/LSTM/GRU/GBDT | Attention/RoPE/LayerNorm/FFN | 统一 OpRegistry，按能力位注册 |
| **执行模式** | 单次执行，固定形状 | 迭代执行（prefill/decode） | 执行器支持两种模式 |
| **批处理** | 固定 batch | 动态 batch / 连续批 | 调度器支持两种策略 |
| **内存模式** | 逐层处理，内存稳定 | KV Cache，随序列增长 | 内存抽象支持两类 |
| **量化需求** | INT8 PTQ/QAT | FP8/KV 量化/激活量化 | 统一 QuantPlugin |
| **精度要求** | FP32/INT8 为主 | FP16/BF16/FP8 为主 | 统一 DType 体系 |
| **典型硬件** | CPU/边缘/GPU | GPU 为主 | 统一硬件画像 |
| **图规模** | 小（数十到数百节点） | 大（数千到数万节点） | 编译优化按规模分档 |
| **瓶颈** | 计算密集（Conv） | 内存带宽密集（Attention/FFN） | 优化策略按瓶颈分档 |

**核心判断**：**统一 IR 和执行器，分离调度策略和内存策略。** 传统模型和 Transformer 共享 90% 的核心，差异集中在 10% 的调度、内存和算子实现上。


## 1. 整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│  接入层   CLI / HTTP / gRPC / C API / Python Binding         │
├─────────────────────────────────────────────────────────────┤
│  前端层   ONNX Loader / TFLite Loader / SafeTensors / GGUF  │
│           Rule DSL / LLM Config / Tool Schema                │
├─────────────────────────────────────────────────────────────┤
│  IR 层    高层语义 IR → 中层执行 IR → 低层后端 IR             │
│           控制流 + 动态维度 + 子图                            │
├─────────────────────────────────────────────────────────────┤
│  编译层   PassManager / 常量折叠 / 算子融合 / 内存规划        │
│           传统模型 Pass / Transformer Pass / 通用 Pass        │
├─────────────────────────────────────────────────────────────┤
│  调度层   Scheduler / BackendScheduler / HybridPlanner       │
│           静态调度器（传统） / 迭代调度器（Transformer）      │
├─────────────────────────────────────────────────────────────┤
│  运行时   Executor / Session / Workspace / EventBus          │
│           单次执行器 / 迭代执行器                             │
├─────────────────────────────────────────────────────────────┤
│  内存层   Allocator / Arena / Pool / TransferManager         │
│           张量池 / KV Cache / 激活池                          │
├─────────────────────────────────────────────────────────────┤
│  硬件层   HardwareDetector / HardwareProfile / CapabilitySet │
│           OptimizationPlanner / KernelVariantRegistry        │
├─────────────────────────────────────────────────────────────┤
│  后端层   CPU Backend + (CUDA | Vulkan) Backend              │
├─────────────────────────────────────────────────────────────┤
│  算子层   通用算子 / 传统算子 / Transformer 算子 / 规则算子   │
├─────────────────────────────────────────────────────────────┤
│  插件层   OpRegistry / BackendRegistry / PassRegistry        │
│           QuantRegistry / SchedulerRegistry / KVCacheRegistry│
│           OptimizerRegistry / ValidatorRegistry              │
├─────────────────────────────────────────────────────────────┤
│  训练层   AutogradEngine / Trainer / Optimizer / Dataloader  │
│           Checkpoint / AMP / RNG（可选，推理时不加载）        │
├─────────────────────────────────────────────────────────────┤
│  校验层   TraceIR / TensorHasher / DeterminismGuard          │
│           Comparator / CrossFrameworkAdapter                 │
├─────────────────────────────────────────────────────────────┤
│  工具层   TraceExporter / GraphExporter / MemoryVisualizer   │
│           ExperimentStore / ComparisonReport / Web UI        │
└─────────────────────────────────────────────────────────────┘

横向：可观测性 | 评测 | 安全 | 版本管理 | 配置 | 特性门控
```

**设备拓扑**：

```text
                ┌──────────────┐
                │  CPU Backend │  （常驻）
                └──────┬───────┘
                       │
              ┌────────┴────────┐
              │  单一 GPU 后端   │
              │  CUDA 或 Vulkan │  （二选一）
              └─────────────────┘
```

**双模型支持策略**：

```text
┌─────────────────────────────────────────────────────────────┐
│                    共享核心（两类模型共用）                    │
│  IR / Graph / Backend / Memory / Tensor / OpRegistry         │
│  KernelRegistry / PluginRegistry / HardwareProfile           │
├─────────────────────────────────────────────────────────────┤
│  传统模型专属                    │  Transformer 专属          │
│  Conv/Pool/BN kernel            │  Attention/RoPE kernel     │
│  LSTM/GRU 执行器                │  KV Cache 管理             │
│  静态调度器                     │  迭代调度器                │
│  固定 batch 处理                │  连续批处理                │
│  INT8 PTQ/QAT                   │  FP8/KV 量化               │
├─────────────────────────────────────────────────────────────┤
│  训练专属（可选）                │  校验专属（可选）          │
│  Autograd / Trainer             │  TraceIR / Comparator      │
└─────────────────────────────────────────────────────────────┘
```


## 2. 硬件能力画像层

> 保持上一版的设计：**优化决策只依赖能力位 + 性能参数，不依赖架构名。**

### 2.1 通用能力画像结构

```cpp
class CapabilitySet {
public:
    bool has(const std::string& name) const;
    void set(const std::string& name, bool value);
    std::vector<std::string> list() const;
private:
    std::unordered_map<std::string, bool> caps_;
};

struct HardwareProfile {
    // ===== 身份（仅用于日志/调试，不参与决策）=====
    std::string vendor;
    std::string device_name;
    std::string driver_version;
    std::string fingerprint;

    // ===== 通用能力位 =====
    CapabilitySet caps;

    // ===== 计算单元 =====
    struct ComputeUnits {
        int num_cores;
        int threads_per_core;
        int max_threads_per_core;
        size_t shared_memory_per_core;
        size_t l2_cache_size;
        size_t l3_cache_size;
        int num_memory_channels;
    } cu;

    // ===== 矩阵/向量加速能力 =====
    struct MatrixUnits {
        bool has_vector_unit;
        bool has_matrix_unit;
        int  matrix_unit_generation;

        // CPU 向量指令集
        bool has_sse2, has_avx, has_avx2;
        bool has_avx512f, has_avx512bw, has_avx512vl;
        bool has_avx512_vnni;
        bool has_avx512_bf16;
        bool has_avx512_fp16;
        bool has_amx_bf16, has_amx_int8, has_amx_fp16;
        bool has_neon, has_sve, has_sve2, has_sme, has_sme2;

        // GPU 矩阵指令集
        bool has_tensor_core;
        bool supports_fp16_mma, supports_bf16_mma, supports_tf32_mma;
        bool supports_int8_mma, supports_int4_mma;
        bool supports_fp8_mma, supports_fp4_mma;
        bool supports_sparsity;

        // 性能参数
        float fp32_tflops, fp16_tflops, bf16_tflops;
        float fp8_tflops, int8_tops, fp4_tflops;
        float tensor_to_vector_ratio;   // ρ
    } matrix;

    // ===== 内存体系 =====
    struct Memory {
        size_t vram_total, vram_available;
        float bandwidth_gbps;
        std::string type;
        int bus_width;
        int memory_levels;
        bool has_error_correction;
    } mem;

    // ===== 数据搬运能力 =====
    struct DataMovement {
        bool supports_async_copy;
        bool supports_bulk_tensor_copy;
        bool supports_p2p;
        bool supports_unified_memory;
        int max_copy_engines;
    } dma;

    // ===== 执行模型特性 =====
    struct ExecutionFeatures {
        bool supports_cooperative_groups;
        bool supports_workgroup_cluster;
        bool supports_distributed_shared_mem;
        bool supports_warp_specialization;
        bool supports_async_pipeline;
        int max_blocks_per_core;
        int max_warps_per_core;
    } exec;

    // ===== 互连拓扑 =====
    struct Interconnect {
        int pcie_gen, pcie_lanes;
        float pcie_bandwidth_gbps;
        bool has_nvlink;
        int nvlink_generation;
        float nvlink_bandwidth_gbps;
        int nvlink_link_count;
        bool has_nvswitch;
    } link;

    // ===== 部署形态 =====
    struct Deployment {
        bool is_mig_enabled;
        int mig_slice_count;
        bool is_vgpu;
        bool is_consumer_grade;
        int tdp_watts;
        std::string power_mode;
        bool is_numa;
        int numa_nodes;
    } deploy;
};
```

### 2.2 硬件探测器

```cpp
class HardwareDetector {
public:
    static Result<HardwareProfile> detect_cpu();
    static Result<HardwareProfile> detect_cuda(int device_id);
    static Result<HardwareProfile> detect_vulkan(int device_id);

private:
    static std::string make_fingerprint(const HardwareProfile&);
};
```

### 2.3 能力驱动的优化决策

```cpp
class OptimizationPlanner {
public:
    struct Plan {
        std::vector<std::string> enabled_passes;
        std::vector<std::string> preferred_kernels;
        MemoryPolicy memory_policy;
        QuantPolicy quant_policy;
        std::string reason;
    };

    static Plan plan(const HardwareProfile& profile,
                     const Graph& graph,
                     const ModelKind kind,        // ===== 新增：模型类别 =====
                     const EngineConfig& config);
};

enum class ModelKind {
    TRADITIONAL,        // CNN / RNN / MLP / GBDT
    TRANSFORMER,        // LLM / ViT / BERT
    HYBRID,             // 混合（如 CNN + Transformer）
    RULE,               // 规则推理
    UNKNOWN
};
```

**模型类别只影响优化策略的选择，不影响 IR 和执行器。** 例如：
- 传统模型：优先 Conv 融合、INT8 量化、固定 batch 调度。
- Transformer：优先 Attention 融合、FP8/KV 量化、连续批处理。
- 混合模型：按子图分类，分别应用对应策略。


## 3. 核心接口（第一天冻结）

### 3.1 KVCachePlugin

```cpp
class KVCachePlugin {
public:
    virtual ~KVCachePlugin() = default;

    virtual Status append(int layer, Tensor k, Tensor v) = 0;
    virtual Result<Tensor> load(int layer, int position) = 0;
    virtual Status clear() = 0;
    virtual size_t memory_usage() const = 0;

    virtual BlockTable block_table() const { return {}; }
    virtual Status evict_blocks(std::span<const BlockId>) { return {}; }
    virtual Status compress(int layer, QuantFormat) { return {}; }
    virtual Result<std::vector<BlockId>> lookup_prefix(BlockHash) { return {}; }
    virtual void register_prefix(BlockHash, std::span<const BlockId>) {}
    virtual Status offload(int layer, int start, int end) { return {}; }
    virtual Status prefetch(int layer, int start, int end) { return {}; }
    virtual Result<TensorHash> hash_state() const { return {}; }

    virtual PluginMeta meta() const = 0;
};

// MVP 实现
class RingKVCache : public KVCachePlugin { ... };

// 未来实现
class PagedKVCache : public KVCachePlugin { ... };
class QuantizedKVCache : public KVCachePlugin { ... };
class HybridKVCache : public KVCachePlugin { ... };
```

### 3.2 Scheduler（双模式）

```cpp
class Scheduler {
public:
    virtual ~Scheduler() = default;

    virtual void submit(Request) = 0;
    virtual BatchPlan plan_next_iteration() = 0;
    virtual Status execute_iteration(const BatchPlan&) = 0;
    virtual void on_request_complete(RequestId) = 0;
    virtual void preempt(RequestId) = 0;

    struct BatchPlan {
        std::vector<PrefillItem> prefills;
        std::vector<DecodeItem> decodes;
        std::vector<TransferItem> transfers;
        size_t token_budget;
    };
};

// 静态调度器（传统模型）
class StaticScheduler : public Scheduler {
    // 固定 batch，单次执行
    // BatchPlan 中只有一次 prefill，无 decode
};

// 迭代调度器（Transformer）
class ContinuousBatchScheduler : public Scheduler {
    // 动态 batch，迭代执行
    // BatchPlan 中包含 prefill + decode
};

// 混合调度器
class HybridScheduler : public Scheduler {
    // 同时处理传统模型和 Transformer 请求
    // 按请求类型路由到不同执行路径
};
```

**关键设计**：`BatchPlan` 结构统一，传统模型只用 `prefills` 一次，Transformer 用 `prefills` + `decodes` 迭代。

### 3.3 Backend

```cpp
class Backend {
public:
    virtual ~Backend() = default;

    virtual Status init() = 0;
    virtual Status shutdown() = 0;
    virtual Device* device() = 0;

    virtual Result<BufferHandle> alloc(size_t) = 0;
    virtual void free(BufferHandle) = 0;
    virtual Status copy_to_device(BufferHandle, const void*, size_t) = 0;
    virtual Status copy_to_host(void*, BufferHandle, size_t) = 0;
    virtual Status copy_d2d(BufferHandle, BufferHandle, size_t) = 0;

    virtual Status compile(Graph&) = 0;
    virtual Status execute_async(const BatchPlan&, ExecutionContext&) = 0;
    virtual Status wait(Timeout) = 0;
    virtual bool is_idle() const = 0;

    virtual BackendCaps caps() const = 0;
    virtual PluginMeta meta() const = 0;

    virtual Result<HardwareProfile> hardware_profile() const = 0;
    virtual Result<MemoryPolicy> memory_policy() const = 0;
};

// MVP 实现（同步包装）
class CpuBackend : public Backend { ... };

// GPU 实现（二选一）
class CudaBackend : public Backend { ... };
class VulkanBackend : public Backend { ... };
```

### 3.4 Graph（支持控制流 + 动态维度 + 训练预留）

```cpp
class Graph {
public:
    std::vector<Node> nodes;
    std::vector<Value> values;
    std::vector<Edge> edges;

    struct DynamicDim {
        ValueId value;
        int dim_index;
        std::string name;
    };
    std::vector<DynamicDim> dynamic_dims;

    // ===== 控制流支持 =====
    struct ControlFlow {
        std::vector<Subgraph> subgraphs;      // if/loop 的子图
        std::vector<LoopInfo> loops;          // 循环信息（RNN 展开）
    };
    ControlFlow control_flow;

    // ===== 训练预留 =====
    bool is_training = false;
    std::optional<Graph> backward_graph;

    Result<BoundGraph> bind(const ShapeMap&) const;
    Result<Graph> build_backward(const std::vector<ValueId>& loss_inputs) const;
    bool is_dynamic() const { return !dynamic_dims.empty(); }
    bool has_control_flow() const { return !control_flow.subgraphs.empty(); }
    bool has_backward() const { return backward_graph.has_value(); }
};

struct LoopInfo {
    NodeId loop_body;
    int64_t max_iterations;
    ValueId condition;
    std::vector<ValueId> carried_values;
};
```

**控制流支持的价值**：
- 传统模型：LSTM/GRU 的循环、条件分支、动态控制流。
- Transformer：通常不需要，但 MoE 路由、早期退出等场景可能用到。

### 3.5 Tensor（训练预留）

```cpp
class Tensor {
public:
    Shape shape;
    DType dtype;
    Device device;
    Layout layout;
    BufferHandle buffer;

    bool requires_grad = false;
    Tensor* grad = nullptr;
    bool is_leaf() const { return grad_fn_ == nullptr; }
    Tensor* grad_fn() const { return grad_fn_.get(); }

private:
    std::shared_ptr<GradFn> grad_fn_;
};
```

### 3.6 Op（支持控制流 + 训练预留）

```cpp
class Op {
public:
    virtual ~Op() = default;

    virtual Status forward(const std::vector<Tensor*>& inputs,
                           std::vector<Tensor*>& outputs,
                           ExecutionContext&) = 0;

    virtual Status backward(const std::vector<Tensor*>& inputs,
                            const std::vector<Tensor*>& outputs,
                            const std::vector<Tensor*>& grad_outputs,
                            std::vector<Tensor*>& grad_inputs,
                            ExecutionContext&) {
        return Status::Unsupported("backward not implemented");
    }

    virtual bool supports_training() const { return false; }

    // ===== 控制流算子预留 =====
    virtual bool is_control_flow() const { return false; }
    virtual Status execute_subgraph(ExecutionContext&,
                                     const Subgraph&) {
        return Status::Unsupported("subgraph not implemented");
    }
};
```

### 3.7 BackendCaps

```cpp
struct BackendCaps {
    bool supports_async = false;
    bool supports_unified_memory = false;
    bool supports_graph_capture = false;

    // ===== 控制流 =====
    bool supports_control_flow = false;

    // ===== 训练预留 =====
    bool supports_training = false;
    bool supports_amp = false;
    bool supports_gradient_checkpointing = false;
    size_t max_grad_buffer_size = 0;

    // ===== 校验预留 =====
    bool supports_deterministic_kernels = false;
    bool supports_batch_invariant = false;

    // ===== 硬件适配 =====
    bool supports_runtime_autotuning = false;
    bool supports_persistent_tuning_cache = false;
};
```


## 4. 核心抽象层

### 4.1 基础类型

```cpp
using Status = std::expected<void, Error>;
template<class T> using Result = std::expected<T, Error>;

enum class Device { CPU, CUDA, VULKAN };
enum class DType {
    F32, F16, BF16, I8, I4, U8,
    FP8_E4M3, FP8_E5M2, MXFP4, MXFP6,
    BOOL  // ===== 新增：控制流条件 =====
};
enum class Layout { ROW_MAJOR, COL_MAJOR, BLOCKED, NHWC, NCHW };

struct Shape {
    std::vector<int64_t> dims;
    int64_t numel() const;
    int64_t stride(int dim) const;
};
```

### 4.2 IR 与计算图

```cpp
enum class OpType {
    // ===== 通用张量算子 =====
    Add, Sub, Mul, Div, MatMul, Relu, Softmax, Concat, Slice, Reshape,
    Transpose, Gather, Scatter, Reduce,

    // ===== 传统模型算子 =====
    Conv2D, DepthwiseConv2D, ConvTranspose2D,
    MaxPool2D, AvgPool2D, GlobalAvgPool,
    BatchNorm, LayerNorm, GroupNorm,
    LSTM, GRU, RNN,
    Upsample, Pad, Clip, Sigmoid, Tanh,
    GBDT_Split, GBDT_Leaf,   // 梯度提升树

    // ===== Transformer 算子 =====
    Attention, MultiHeadAttention, RoPE, RMSNorm,
    Embedding, PositionalEncoding,
    MoE_Router, MoE_Expert,

    // ===== 控制流 =====
    If, Loop, While, Subgraph,

    // ===== 规则算子 =====
    RuleMatch, RuleFire, FactAssert,

    // ===== 编排算子 =====
    ToolCall, LLMGenerate, Route, Merge,

    // ===== 训练预留 =====
    CrossEntropy, MSE, Dropout, SoftmaxCrossEntropy
};

struct Node {
    NodeId id;
    OpType op;
    std::vector<ValueId> inputs, outputs;
    Attrs attrs;
    std::optional<Device> placed_device;
    ModelKind kind = ModelKind::UNKNOWN;   // ===== 新增：节点归属类别 =====
};

class OpRegistry {
public:
    using Creator = std::function<std::unique_ptr<Op>()>;
    void register_op(OpType, Creator);
    Result<std::unique_ptr<Op>> create(OpType);

    // ===== 按模型类别查询 =====
    std::vector<OpType> list_by_kind(ModelKind kind);
};
```

### 4.3 内存抽象

```cpp
enum class MemoryKind {
    HOST, DEVICE, PINNED, UNIFIED, MANAGED,
    GRADIENT, OPTIMIZER_STATE, ACTIVATION,
    // ===== 传统模型专属 =====
    WEIGHT_CONST,          // 常量权重
    // ===== Transformer 专属 =====
    KV_CACHE               // KV Cache
};

class Allocator {
public:
    virtual Result<BufferHandle> alloc(size_t, MemoryKind) = 0;
    virtual void free(BufferHandle) = 0;
    virtual size_t available(MemoryKind) const = 0;
    virtual Status free_gradients() { return {}; }
    virtual Status free_activations() { return {}; }
};

class Arena { };
class MemoryPool { };

// ===== 内存策略（按模型类别分档）=====
struct MemoryPolicy {
    // 通用
    size_t tensor_pool_size;
    size_t activation_pool_size;

    // 传统模型
    size_t weight_cache_size;      // 权重常驻内存
    bool enable_weight_prefetch;   // 逐层预取

    // Transformer
    size_t kv_cache_block_size;
    bool enable_prefix_caching;
    size_t max_concurrent_requests;
};
```

### 4.4 插件框架

```cpp
struct PluginMeta {
    std::string name, version, api_version;
    std::vector<std::string> tags;   // "traditional" | "transformer" | "quant" | ...
    std::string description;
};

template<class PluginT>
class PluginRegistry {
public:
    static PluginRegistry& instance();
    void register_plugin(std::unique_ptr<PluginT>);
    Result<PluginT*> get(const std::string& name);
    std::vector<PluginMeta> list() const;
    std::vector<PluginMeta> list_by_tag(const std::string& tag) const;
    Status load_shared_library(const std::string& path);
};
```

### 4.5 RNG 抽象

```cpp
class RNG {
public:
    virtual ~RNG() = default;
    virtual uint64_t next_u64() = 0;
    virtual float next_float() = 0;
    virtual void set_seed(uint64_t) = 0;
    virtual Status save_state(BufferHandle) = 0;
    virtual Status load_state(BufferHandle) = 0;
};

class SamplingRNG : public RNG { ... };
class TrainingRNG : public RNG { ... };

class RNGRegistry {
public:
    static RNGRegistry& instance();
    RNG* get(const std::string& name);
};
```


## 5. 后端层

### 5.1 设备选择策略

```cpp
enum class GpuBackend { NONE, CUDA, VULKAN };

class DeviceSelector {
public:
    static GpuBackend select();
};

struct EngineConfig {
    GpuBackend gpu_backend = GpuBackend::NONE;
    bool enable_cpu = true;
    bool enable_training = false;
    bool enable_validation = false;
    bool enable_deterministic = false;

    // ===== 模型类别 =====
    ModelKind default_model_kind = ModelKind::UNKNOWN;

    struct HardwareAdaptation {
        bool auto_detect = true;
        bool enable_runtime_autotuning = true;
        bool persist_tuning_cache = true;
        std::string tuning_cache_dir;
        bool enable_capability_driven_selection = true;
        int autotune_budget_ms = 100;
    } hardware_adapt;

    HybridConfig hybrid;
    ValidationConfig validation;
};

Status validate(const EngineConfig& cfg);
```

### 5.2 CPU 后端（能力位驱动）

```cpp
class CpuBackend : public Backend {
    HardwareProfile profile_;
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<CpuAllocator> allocator_;
};
```

**算子变体（按能力位组织）**：

| 算子类别 | 变体 |
|---|---|
| MatMul | scalar / sse2 / avx2 / avx512 / avx512_vnni / amx_bf16 / amx_int8 / amx_fp16 |
| Conv2D | scalar / im2col+matmul / direct / winograd / fft |
| Pool | scalar / sse2 / avx2 / avx512 |
| Attention | naive / flash / paged |
| LSTM/GRU | scalar / avx2 / avx512 / amx |

### 5.3 CUDA 后端（能力位驱动）

| 能力组合 | 策略 |
|---|---|
| `supports_fp8_mma && supports_bulk_tensor_copy` | TMA + FP8 原生路径 |
| `supports_fp8_mma && !supports_bulk_tensor_copy` | FP8 原生路径（无 TMA） |
| `!supports_fp8_mma && ρ > 32` | FP16/BF16 Tensor Core 激进路径 |
| `!supports_fp8_mma && ρ < 16` | Tensor Core + CUDA Core 混合 |
| `supports_fp4_mma` | FP4 路径 |

### 5.4 Vulkan 后端（能力位驱动）

- **Kernel**：GLSL compute shader → SPIR-V → `vkCmdDispatch`
- **矩阵单元**：`VK_KHR_cooperative_matrix`
- **Bulk copy**：`VK_KHR_copy_commands2`
- **内存**：VMA + staging buffer

### 5.5 后端能力矩阵

```cpp
struct BackendMatrix {
    std::unordered_map<OpType, std::array<bool, 3>> op_support;
    std::unordered_map<DType, std::array<bool, 3>> dtype_support;
};
```


## 6. 自适应调优层

### 6.1 三层自适应机制

```text
┌─────────────────────────────────────────────────────────┐
│              自适应调优三层架构                           │
├─────────────────────────────────────────────────────────┤
│  第 1 层：编译期能力分档                                  │
│  按能力位组合预编译 kernel 变体                          │
├─────────────────────────────────────────────────────────┤
│  第 2 层：启动期选择                                      │
│  根据 HardwareProfile 能力位筛选变体                     │
├─────────────────────────────────────────────────────────┤
│  第 3 层：运行期微调                                      │
│  对未覆盖的 shape 组合进行在线 benchmark 并缓存           │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Kernel 变体定义（能力前置条件）

```cpp
struct KernelVariant {
    std::string name;
    ModelKind target_kind = ModelKind::UNKNOWN;   // ===== 新增 =====

    struct Requirements {
        std::vector<std::string> required_caps;
        std::vector<std::string> forbidden_caps;
        float min_tensor_to_vector_ratio = 0.0f;
        float max_tensor_to_vector_ratio = 1e9f;
        size_t min_vram = 0;
        float min_bandwidth = 0.0f;
        size_t min_l3_cache = 0;
    } req;

    struct CostModel {
        float estimated_tflops;
        float estimated_bandwidth_gbps;
        float estimated_latency_us;
    } cost;
};

class KernelVariantRegistry {
public:
    void register_variant(OpType op, DType dtype, KernelVariant variant);
    Result<std::vector<KernelVariant>> select(
        OpType op, DType dtype, const Shape& shape,
        const HardwareProfile& profile,
        ModelKind kind = ModelKind::UNKNOWN);
};
```

### 6.3 运行期微调

```cpp
class RuntimeAutotuner {
public:
    struct TunerConfig {
        bool enabled = true;
        int max_tuning_iterations = 3;
        double tuning_budget_ms = 100.0;
        bool persist_cache = true;
        std::string cache_dir;
    };

    Result<Kernel*> tune(OpType op, DType dtype, const Shape& shape,
                          const HardwareProfile& profile,
                          ModelKind kind);
};

class HardwareAwareCache {
public:
    struct Key {
        std::string hardware_fingerprint;
        OpType op;
        DType dtype;
        Shape shape;
        ModelKind kind;
    };
    Result<Value> get(const Key&);
    Status put(const Key&, const Value&);
    Status save(const std::string& path);
    Status load(const std::string& path);
};
```

### 6.4 运行期硬件画像细化

```cpp
class RuntimeHardwareProfiler {
public:
    struct MeasuredPerformance {
        float achieved_tflops;
        float achieved_bandwidth_gbps;
        float achieved_latency_us;
        float power_draw_watts;
        float temperature_c;
        float clock_mhz;
        bool is_throttling;
    };

    void update(const MeasuredPerformance&);
    OptimizationPlanner::Plan replan(const HardwareProfile& profile,
                                      const MeasuredPerformance& perf,
                                      ModelKind kind);
};
```


## 7. 混合推理层（两设备协同）

### 7.1 切分策略

| 策略 | 适用 | 并行性 |
|---|---|---|
| 层间分割 | 通用 | 串行，利用率低 |
| 专家分割 | MoE | 真正并行，首选 |
| 神经元分割 | 密集 LLM | 细粒度，需 profiling |
| 流水线并行 | 通用 | 计算与传输重叠 |
| 投机解码 | Transformer | 异构各司其职 |
| 算子级分割 | 通用图 | 按算子类型分配 |
| **Conv/MatMul 分割** | **传统模型** | **按算子类型分配** |

### 7.2 配置接口

```cpp
struct HybridConfig {
    enum class Strategy {
        AUTO, LAYER_WISE, EXPERT_WISE, NEURON_WISE,
        PIPELINE, SPECULATIVE, OP_LEVEL
    } strategy = Strategy::AUTO;

    struct DeviceConfig {
        BackendKind backend;
        size_t memory_budget;
        float compute_share;
    };
    std::vector<DeviceConfig> devices;

    struct LayerPlacement { int layer_start, layer_end; BackendKind target; };
    std::vector<LayerPlacement> manual_placement;

    struct ExpertConfig {
        int max_experts_on_gpu;
        float hotness_threshold;
        bool enable_prefetch;
    } expert_cfg;

    struct KVCacheConfig {
        size_t gpu_kv_budget;
        bool offload_to_cpu;
        size_t cpu_kv_budget;
    } kv_cfg;

    struct PipelineConfig { bool enable_prefetch; int prefetch_depth; } pipeline_cfg;
};
```

### 7.3 混合规划器

```cpp
class HybridPlanner {
public:
    struct NodePlacement {
        NodeId node;
        BackendKind backend;
        float estimated_cost;
        size_t memory_footprint;
        bool needs_transfer;
    };

    Result<std::vector<NodePlacement>> plan(const Graph&, const HybridConfig&);

private:
    std::vector<Device> candidate_devices() const;
    void analyze_op_types(Graph&);
    void classify_model_kind(Graph&);   // ===== 新增：标注节点类别 =====
    NodePlacement decide_placement(const Node&, const HybridConfig&);
    void balance_load(std::vector<NodePlacement>&);
    void insert_transfers(Graph&, const std::vector<NodePlacement>&);
};
```

### 7.4 混合调度器

```cpp
class HybridScheduler {
public:
    Status execute(const Graph&, const ExecutionPlan&, ExecutionContext&);

private:
    std::unique_ptr<CpuExecutor> cpu_exec_;
    std::unique_ptr<GpuExecutor> gpu_exec_;
    EventBus event_bus_;
    TransferManager transfer_mgr_;
    DeviceQueue cpu_queue_, gpu_queue_;
};
```

### 7.5 跨设备内存与传输

```cpp
class TransferManager {
public:
    void register_route(Device from, Device to, TransferFn);
    Status transfer(BufferHandle dst, Device dst_dev,
                    BufferHandle src, Device src_dev);
    Status prefetch_async(BufferHandle dst, const void* src, Stream);
    Status wait(TransferHandle);
};
```

**传输路径矩阵**：

| From \ To | CPU | CUDA | Vulkan |
|---|---|---|---|
| CPU | memcpy | cudaMemcpyAsync | staging + vkCmdCopyBuffer |
| CUDA | cudaMemcpyAsync | cudaMemcpyAsync | — |
| Vulkan | staging + vkMapMemory | — | vkCmdCopyBuffer |


## 8. 推理范式层

### 8.1 规则推理

```cpp
struct Fact { std::string type; std::unordered_map<std::string, Value> fields; };
struct Condition { std::function<bool(const Fact&, const WorkingMemory&)> eval; };
struct Rule {
    std::string name;
    Condition condition;
    std::function<void(WorkingMemory&)> action;
    int priority;
};

class RuleEngine {
public:
    void add_fact(Fact);
    void add_rule(Rule);
    Result<Agenda> fire();
    const Trace& trace() const;
};
```

### 8.2 传统模型推理

```cpp
class TraditionalEngine {
public:
    // 加载 ONNX / TFLite / 自定义格式
    Result<void> load_model(const std::string& path, ModelFormat format);
    Result<Tensor> infer(const std::vector<Tensor>& inputs);

    // 批量推理
    Result<std::vector<Tensor>> infer_batch(
        const std::vector<std::vector<Tensor>>& batch);
};

enum class ModelFormat {
    ONNX, TFLITE, CAFFE, SAFETENSORS, GGUF, CUSTOM
};
```

### 8.3 Transformer 推理

```cpp
class Tokenizer {
public:
    virtual Result<std::vector<int>> encode(std::string_view) = 0;
    virtual Result<std::string> decode(std::span<const int>) = 0;
};

class Sampler {
public:
    virtual int sample(std::span<const float> logits, const SamplingParams&) = 0;
};

class LLMEngine {
public:
    virtual Status load(const ModelConfig&) = 0;
    virtual Status generate(std::string_view prompt,
                            const GenerationParams&,
                            std::function<void(std::string_view)> on_token) = 0;
};
```

### 8.4 混合编排

```cpp
class HybridOrchestrator {
public:
    Result<Graph> build_pipeline(const PipelineSpec&);
    Result<Output> run(const Graph&, const Input&);
};
```


## 9. 训练层（可选加载）

### 9.1 训练插件类型

```cpp
class OptimizerPlugin {
public:
    virtual PluginMeta meta() const = 0;
    virtual std::string optimizer_name() const = 0;
    virtual Status step(std::vector<Tensor*>& params,
                        std::vector<Tensor*>& grads,
                        const OptimizerConfig&) = 0;
    virtual Status zero_grad(std::vector<Tensor*>& grads) = 0;
    virtual Status save_state(BufferHandle) = 0;
    virtual Status load_state(BufferHandle) = 0;
};

class LRSchedulerPlugin {
public:
    virtual PluginMeta meta() const = 0;
    virtual std::string scheduler_name() const = 0;
    virtual float get_lr(int step) = 0;
};

class LossPlugin {
public:
    virtual PluginMeta meta() const = 0;
    virtual std::string loss_name() const = 0;
    virtual Tensor compute(const Tensor& pred, const Tensor& target) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
};

class DataloaderPlugin {
public:
    virtual PluginMeta meta() const = 0;
    virtual Status load(const std::string& path) = 0;
    virtual Result<Batch> next() = 0;
    virtual void shuffle(uint64_t seed) = 0;
};
```

### 9.2 AutogradEngine

```cpp
class AutogradEngine {
public:
    void record(const Op& op,
                const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs);
    Status backward(Tensor& loss);
    Result<Tensor*> grad(const Tensor&);
    void zero_grad();
    void set_training(bool);
    bool is_training() const;
};
```

### 9.3 Trainer

```cpp
class Trainer {
public:
    struct Config {
        int epochs = 10;
        int batch_size = 32;
        float lr = 1e-4;
        std::string optimizer = "adamw";
        std::string scheduler = "cosine";
        std::string loss = "cross_entropy";
        bool use_amp = false;
        int grad_accum_steps = 1;
        int checkpoint_interval = 1000;
        std::string checkpoint_dir;
    };

    Status train(Graph& model, Dataloader& dataloader, const Config&);
    Result<Metrics> evaluate(Graph& model, Dataloader&);
    Status resume(const std::string& checkpoint_path);
};
```

### 9.4 Checkpoint 格式

```cpp
struct Checkpoint {
    std::unordered_map<std::string, Tensor> model_weights;

    std::optional<std::unordered_map<std::string, Tensor>> optimizer_state;
    std::optional<std::unordered_map<std::string, Tensor>> scheduler_state;
    std::optional<std::unordered_map<std::string, BufferHandle>> rng_state;
    std::optional<int> global_step;
    std::optional<int> epoch;
    std::optional<Graph> graph;
};

class CheckpointIO {
public:
    static Status save(const Checkpoint&, const std::string& path);
    static Result<Checkpoint> load(const std::string& path);
    static Result<std::unordered_map<std::string, Tensor>>
        load_weights_only(const std::string& path);
};
```

### 9.5 训练模块目录

```text
train/
  autograd/ optimizer/ scheduler/ loss/ data/ trainer/ amp/ checkpoint/ distributed/
```


## 10. 校验与调试层（可选加载）

### 10.1 四层校验管线

```text
第 1 层  任务级校验 (Task-level)
第 2 层  Prompt 级校验 (Prompt-level)
第 3 层  张量级校验 (Tensor-level)
第 4 层  权重级校验 (Weight-level)
```

### 10.2 参考追踪 (ReferenceTracer)

```cpp
struct TraceIR {
    struct TensorRecord {
        std::string name;
        Shape shape;
        DType dtype;
        std::vector<uint8_t> data;
        std::string op_type;
        int64_t timestamp;
    };

    struct OpRecord {
        std::string op_name;
        std::string op_type;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        Attrs attrs;
    };

    std::vector<TensorRecord> tensors;
    std::vector<OpRecord> ops;
    ModelMeta model_meta;
};

class ReferenceTracer {
public:
    Result<TraceIR> trace_pytorch(const std::string& model_path,
                                   const std::vector<Tensor>& inputs);
    Result<TraceIR> trace_onnx(const std::string& model_path,
                                const std::vector<Tensor>& inputs);
    Result<TraceIR> load(const std::string& path);
    Status save(const TraceIR&, const std::string& path);
};
```

### 10.3 张量哈希 (TensorHasher)

```cpp
class TensorHasher {
public:
    enum class HashType { XXH64, SHA1, SHA256, UUID };

    static Result<std::string> hash_tensor(const Tensor&, HashType = HashType::XXH64);
    static Result<HashManifest> hash_model(const Graph&, HashType);
    static Result<VerifyResult> verify_manifest(const Graph&, const HashManifest&);
};

struct HashManifest {
    std::string model_hash;
    std::unordered_map<std::string, std::string> tensor_hashes;
    HashType hash_type;
};

struct VerifyResult {
    bool passed;
    std::vector<std::string> mismatched_tensors;
    std::string first_mismatch;
};
```

### 10.4 确定性保障 (DeterminismGuard)

```cpp
struct DeterminismConfig {
    bool enabled = false;
    bool deterministic_kernels = false;
    bool batch_invariant = false;
    bool disable_tf32 = true;
    bool use_ieee_rounding = true;
    bool disable_prefix_caching = true;
};

class DeterminismGuard {
public:
    Status enable(const DeterminismConfig&);
    Result<BatchInvarianceReport> verify_batch_invariance(
        const std::string& prompt, int64_t seed = 42);
    Status warmup();
};
```

### 10.5 张量级对比器 (Comparator)

```cpp
class Comparator {
public:
    struct CompareConfig {
        double rtol = 1e-3;
        double atol = 1e-5;
        bool compare_first_token = true;
        int max_divergence_points = 10;
        bool compare_kv_cache = true;
    };

    static Result<CompareReport> compare(
        const TraceIR& reference, const TraceIR& candidate,
        const CompareConfig&);

    static Result<TensorDiff> compare_tensor(
        const Tensor& ref, const Tensor& cand, const CompareConfig&);

    static Result<DivergencePoint> find_first_divergence(
        const TraceIR& reference, const TraceIR& candidate);
};
```

### 10.6 跨框架适配器 (CrossFrameworkAdapter)

```cpp
class CrossFrameworkAdapter {
public:
    static Result<TraceIR> to_trace_ir(const EngineOutput&);
    static Result<TraceIR> from_vllm(const std::string& output_dir);
    static Result<TraceIR> from_tensorrt_llm(const std::string& output_dir);
    static Result<TraceIR> from_huggingface(const std::string& output_dir);
    static Result<TraceIR> from_onnxruntime(const std::string& output_dir);

    static Result<CompareReport> full_pipeline(
        const std::string& engine_output,
        const std::string& reference_output,
        const CompareConfig&);
};
```

### 10.7 校验配置

```cpp
struct ValidationConfig {
    bool enabled = false;

    struct TracerConfig {
        bool enable_pytorch_trace = false;
        bool enable_onnx_trace = false;
        std::string trace_output_dir;
        bool capture_kv_cache = false;
    } tracer;

    struct HashConfig {
        bool enable = false;
        TensorHasher::HashType hash_type = TensorHasher::HashType::XXH64;
        bool per_tensor_layer = true;
        std::string manifest_path;
    } hasher;

    DeterminismConfig determinism;
    Comparator::CompareConfig comparator;

    struct CrossFrameworkConfig {
        bool enable = false;
        std::string reference_framework;
        std::string reference_output_dir;
    } cross_framework;
};
```

### 10.8 校验模块目录

```text
validation/
  reference_tracer/ tensor_hash/ determinism/ comparator/ cross_framework/ report/
```


## 11. 模型转换层

### 11.1 支持的模型格式

| 格式 | 主要生态 | 模型类型 |
|---|---|---|
| ONNX | ONNX Runtime / Triton / OpenVINO | 传统 + Transformer |
| TFLite | TensorFlow Lite / 移动端 | 传统 |
| Caffe | 经典 CV | 传统 |
| Safetensors | HuggingFace | Transformer |
| GGUF | llama.cpp / Ollama | Transformer |
| TensorRT Engine | TensorRT-LLM | Transformer |

### 11.2 ModelConverter 接口

```cpp
class ModelConverter {
public:
    // 传统模型
    static Result<void> convert_onnx_to_ie(
        const std::string& onnx_path, const std::string& output_path);
    static Result<void> convert_tflite_to_ie(
        const std::string& tflite_path, const std::string& output_path);

    // Transformer 模型
    static Result<void> convert_hf_to_safetensors(
        const std::string& hf_path, const std::string& output_path);
    static Result<void> convert_hf_to_gguf(
        const std::string& hf_path, const std::string& output_path,
        GGUFQuantType quant = GGUFQuantType::F16);

    // 统一加载入口
    static Result<Graph> load_model(const std::string& path,
                                     const ModelLoaderConfig& config);
};

enum class GGUFQuantType {
    F32, F16, BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q3_K_M, Q2_K
};
```

### 11.3 五阶段转换验证

```cpp
class ConversionValidator {
public:
    struct ValidationConfig {
        double rtol = 1e-3;
        double atol = 1e-5;
        bool layer_by_layer = true;
        bool check_tokenizer = true;
        bool check_metadata = true;
    };

    struct ValidationResult {
        bool passed;
        std::string stage;
        std::string detail;
    };

    static Result<ValidationResult> validate_structure(
        const Graph& source, const Graph& target);
    static Result<ValidationResult> validate_weights(
        const std::string& source_path, const std::string& target_path);
    static Result<ValidationResult> validate_layer_outputs(
        const std::string& source_path, const std::string& target_path,
        const std::vector<Tensor>& inputs);
    static Result<ValidationResult> validate_end_to_end(
        const std::string& source_path, const std::string& target_path,
        const std::vector<Tensor>& inputs);
    static Result<ValidationResult> validate_tokenizer_metadata(
        const std::string& source_path, const std::string& target_path);
};
```

### 11.4 目录结构

```text
converter/
  model_converter.h
  format_loader/
    onnx_loader.h        # 传统模型
    tflite_loader.h      # 传统模型
    caffe_loader.h       # 传统模型
    safetensors_loader.h # Transformer
    gguf_loader.h        # Transformer
  op_mapping/
    onnx_mapping.h
    tflite_mapping.h
    pytorch_mapping.h
    gguf_mapping.h
  validation/
    conversion_validator.h
    weight_comparator.h
  quantization/
    int8_quant.h         # 传统模型
    gguf_quant.h         # Transformer
    awq_quant.h
    gptq_quant.h
```


## 12. 可演进架构（前沿接入）

### 12.1 插件类型

| 插件类型 | 接口 | 接入内容 |
|---|---|---|
| `OpPlugin` | `create(Attrs)` | Mamba、SSM、RWKV、新 Conv 变体 |
| `BackendPlugin` | `create(Config)` | NPU、TPU、ANE |
| `PassPlugin` | `run(Graph&, Context&)` | 新编译优化 |
| `QuantPlugin` | `dequantize / quantize` | FP8、MXFP4、AWQ、GPTQ、INT8 PTQ |
| `SchedulerPlugin` | `plan(Graph&, Context&)` | 连续批、Chunked Prefill |
| `KVCachePlugin` | `append / load / evict` | Paged、Quantized、Hybrid |
| `SamplerPlugin` | `sample(logits, params)` | 投机解码、Medusa、EAGLE |
| `ModelLoaderPlugin` | `load(config)` | 新模型格式 |
| `TokenizerPlugin` | `encode / decode` | 新分词器 |
| `OptimizerPlugin` | `step / zero_grad` | SGD、Adam、AdamW |
| `LRSchedulerPlugin` | `get_lr` | Cosine、Linear |
| `LossPlugin` | `compute / backward` | CrossEntropy、MSE |
| `DataloaderPlugin` | `load / next` | 数据集加载 |
| `HardwareProbePlugin` | `probe()` | 新硬件探测 |

### 12.2 前沿技术接入点

| 方向 | 接入方式 | 预留接口 | TODO |
|---|---|---|---|
| 量化 | `QuantPlugin` | dequant/quant | FP8、MXFP4、AWQ、GPTQ、GGUF、INT8 PTQ |
| 注意力 | `OpPlugin` | AttentionOp 变体 | FlashAttention-2/3、PagedAttention、MLA |
| MoE | `OpPlugin`+`SchedulerPlugin` | MoEExpertOp | DeepSeek MoE、Mixtral |
| KV Cache | `KVCachePlugin` | append/load/evict/compress | H2O、StreamingLLM、KV 量化 |
| 采样 | `SamplerPlugin` | sample(logits, params) | 投机解码、Medusa、EAGLE |
| 编译优化 | `PassPlugin` | run(Graph&, Context&) | 融合、布局转换、图重写 |
| 调度 | `SchedulerPlugin` | plan(Graph&, Context&) | 连续批、Chunked Prefill、PD 分离 |
| 并行 | `BackendPlugin`+`PlannerPlugin` | TP/PP/EP/SP | 张量/流水/专家/序列并行 |
| 异构 | `PlannerPlugin` | plan(Graph&, HybridConfig&) | 层间/专家/神经元/投机/流水线 |
| 结构化输出 | `OpPlugin` | ConstraintDecodeOp | JSON Schema、正则、FSM |
| 工具调用 | `OpPlugin` | ToolCallOp | Function Calling、MCP、ReAct |
| 多模态 | `OpPlugin` | VisionEncoderOp、AudioEncoderOp | ViT、CLIP、Whisper |
| 新架构 | `ModelLoaderPlugin` | load(config) | Mamba、RWKV、RetNet、Hyena |
| 分布式 | `BackendPlugin` | 远程后端 | Ray、NCCL、gRPC |
| 训练优化 | `OptimizerPlugin` | step / zero_grad | ZeRO、FSDP、LoRA |
| 混合精度 | `PassPlugin` | run(Graph&, Context&) | AMP、BF16、FP8 训练 |
| 确定性推理 | `BackendCaps` | `supports_batch_invariant` | 确定性 Kernel、批次不变性 |
| 新硬件 | `HardwareProbePlugin` | `probe()` | 新厂商能力位归一化 |

### 12.3 特性门控

```cpp
struct FeatureFlags {
    bool enable_experimental = false;
    bool enable_fp8 = false;
    bool enable_speculative = false;
    bool enable_paged_attention = false;
    bool enable_moe = false;
    bool enable_training = false;
    bool enable_amp = false;
    bool enable_distributed = false;
    bool enable_validation = false;
    bool enable_deterministic = false;
    bool enable_reference_trace = false;
    bool enable_capability_driven_selection = true;
    bool enable_runtime_autotuning = true;
    // ===== 传统模型 =====
    bool enable_conv_winograd = true;
    bool enable_conv_fft = false;
    bool enable_int8_ptq = true;
    // ===== Transformer =====
    bool enable_flash_attention = true;
    bool enable_kv_cache_quant = true;
};
```

### 12.4 API 版本与 ABI

```cpp
#define IE_API_VERSION_MAJOR 0
#define IE_API_VERSION_MINOR 1

extern "C" {
    IE_API Backend* ie_create_backend(const char* kind, const char* config);
    IE_API void ie_destroy_backend(Backend*);
    IE_API const char* ie_api_version();
}
```


## 13. 可视化与对比工具链

### 13.1 数据采集层

```cpp
class TraceExporter {
public:
    void begin_event(const std::string& name, const std::string& category);
    void end_event();
    void instant_event(const std::string& name, const std::string& category);
    void counter(const std::string& name, double value);
    void set_device(Device);
    void set_stream(int stream_id);
    void set_phase(const std::string& phase);
    void set_model_kind(ModelKind);   // ===== 新增 =====
    Status flush(const std::string& path);
};

class GraphExporter {
public:
    static Status to_dot(const Graph&, const std::string& path);
    static Status to_json(const Graph&, const std::string& path);
    static Status to_mermaid(const Graph&, const std::string& path);
};

class MetricsExporter {
public:
    Status to_prometheus(const std::string& path);
    Status to_json(const std::string& path);
};
```

### 13.2 可视化层

| 可视化 | 功能 | 实现路径 |
|---|---|---|
| 计算图 | DAG、设备着色、模型类别着色、融合前后对比 | Graph → JSON/DOT → Web UI |
| 调度时间线 | 多设备时间线、跨设备依赖、kernel 级 | Chrome Trace → Perfetto |
| 内存 | 张量生命周期、池统计、KV Cache | 甘特图 → Web UI |
| 规则推理 | 规则触发链、议程演化 | JSON → Mermaid sequence |
| LLM | Token 时间线、Attention 热力图、KV Cache 状态 | 自定义 → Web UI |
| 性能 Dashboard | 实时指标、历史趋势 | Prometheus → Grafana |
| 训练 | Loss 曲线、梯度范数、学习率 | 自定义 → Web UI |
| 校验 | 张量分歧热力图、逐层误差曲线 | HTML 报告 |
| 硬件画像 | 能力位雷达图、性能参数量级 | Web UI |
| 调优决策 | kernel 选择依据、ρ 值、缓存命中率 | Web UI |
| **传统模型** | **Conv 层时间线、算子级 profiling** | **Web UI** |

### 13.3 对比框架

```cpp
struct Experiment {
    std::string id, name;
    struct Config {
        std::vector<BackendKind> devices;
        HybridConfig hybrid;
        std::vector<std::string> passes;
        std::string quant_format;
        std::string scheduler_policy;
        std::string kv_cache_policy;
        ModelKind model_kind = ModelKind::UNKNOWN;
        bool is_training = false;
        std::string optimizer;
        std::string lr_scheduler;
        std::string loss;
        bool use_amp = false;
        bool enable_validation = false;
        bool enable_deterministic = false;
        FeatureFlags features;
        std::unordered_map<std::string, std::string> extras;
    } config;
    std::string graph_path, input_path;
    HardwareProfile hardware_profile;

    struct Result {
        double latency_p50, latency_p95, latency_p99;
        double throughput;
        size_t memory_peak;
        double accuracy;
        double final_loss = 0;
        double best_metric = 0;
        int total_steps = 0;
        bool validation_passed = false;
        double max_divergence = 0;
        std::string trace_path, metrics_path;
    } result;
};

class ExperimentStore {
public:
    void save(const Experiment&);
    Result<Experiment> load(const std::string& id);
    std::vector<Experiment> list(const std::string& filter);
    Status export_csv(const std::string& path);
    Status export_json(const std::string& path);
    std::vector<Experiment> find_by_capability(const std::string& cap_name, bool value);
    std::vector<Experiment> find_by_bandwidth_range(float min, float max);
    std::vector<Experiment> find_by_model_kind(ModelKind kind);
};

class ExperimentRunner {
public:
    struct Matrix {
        std::vector<std::vector<BackendKind>> device_sets = {
            {BackendKind::CPU},
            {BackendKind::CPU, BackendKind::CUDA},
            {BackendKind::CPU, BackendKind::VULKAN},
        };
        std::vector<HybridConfig::Strategy> strategies;
        std::vector<std::string> quant_formats;
        std::vector<std::string> scheduler_policies;
        std::vector<std::string> kv_cache_policies;
        std::vector<std::string> optimizers;
        std::vector<std::string> lr_schedulers;
        std::vector<ModelKind> model_kinds;   // ===== 新增 =====
    };
    Result<std::vector<Experiment>> run_matrix(const Matrix&, const Graph&);
    Result<std::vector<Experiment>> run_ablation(
        const Experiment::Config& base,
        const std::string& variable,
        const std::vector<std::string>& values);
    Result<std::vector<Experiment>> run_cross_hardware(
        const std::vector<HardwareProfile>& profiles,
        const Graph&);
};

class ComparisonReport {
public:
    static Status to_markdown(const std::vector<Experiment>&, const std::string& path);
    static Status to_html(const std::vector<Experiment>&, const std::string& path);
    static Status to_csv(const std::vector<Experiment>&, const std::string& path);
    static std::vector<Diff> compare(const Experiment& baseline,
                                     const Experiment& candidate);
    static Status to_cross_hardware_report(
        const std::vector<Experiment>&, const std::string& path);
};
```

### 13.4 对比场景

| 场景 | 变量 | 对比指标 |
|---|---|---|
| 后端对比 | CPU-only / CPU+CUDA / CPU+Vulkan | 延迟、吞吐、内存、数值误差 |
| 量化对比 | FP16 / Q8 / Q4_K_M / AWQ / GPTQ / FP8 / INT8 | 延迟、显存、精度 |
| 混合策略 | CPU+CUDA × {层间/专家/神经元/流水线/投机} | 延迟、GPU 利用率、PCIe 量、加速比 |
| 调度策略 | Simple / 连续批 / Chunked / 混合 | QPS、P99、利用率、批大小 |
| KV Cache | Ring / Paged / Quantized / Hybrid | 显存、命中率、TTFT |
| 编译优化 | 无 / 常量折叠 / 融合 / 全优化 | 算子数、launch 数、延迟 |
| 训练优化器 | SGD / Adam / AdamW | 收敛速度、最终 loss、显存 |
| 框架对齐 | 本引擎 vs vLLM vs TensorRT-LLM vs ONNX Runtime | 张量误差、首个分歧点 |
| 确定性验证 | 确定性模式开/关 | 批次不变性通过率 |
| 跨硬件对比 | A100 vs 3050 vs H200 vs i7 vs Xeon | 同代码在不同能力位的性能/精度 |
| CPU 指令集对比 | AVX2 vs AVX-512 vs VNNI vs AMX | MatMul/Conv 吞吐 |
| **传统模型对比** | **ResNet / MobileNet / YOLO / BERT** | **延迟、吞吐、精度** |
| **Transformer 对比** | **Llama / Qwen / Mistral** | **延迟、吞吐、TTFT、TPOT** |

### 13.5 可复现性

```json
{
  "experiment_id": "exp_20250101_120000_abc",
  "timestamp": "2025-01-01T12:00:00Z",
  "git_commit": "abc123",
  "build_type": "Release",
  "compiler": "gcc 13.2",
  "model_kind": "TRANSFORMER",
  "hardware_profile": {
    "vendor": "nvidia",
    "device_name": "NVIDIA H200",
    "fingerprint": "fp_abc123",
    "caps": {
      "supports_fp8_mma": true,
      "supports_bulk_tensor_copy": true,
      "supports_workgroup_cluster": true
    },
    "matrix": { "tensor_to_vector_ratio": 32.0, "fp8_tflops": 1979.0 },
    "mem": { "bandwidth_gbps": 4800.0, "type": "HBM3e" }
  },
  "config": { ... },
  "input_hash": "sha256:...",
  "result": { ... }
}
```

### 13.6 CLI 工具

```bash
# 推理
ie-run    --config exp.yaml --graph model.json --output results/
ie-run    --matrix matrix.yaml --output results/
ie-report --input results/ --format html --output report.html
ie-trace  --config exp.yaml --output trace.json
ie-graph  --graph model.json --format dot --output graph.dot

# 传统模型
ie-run    --config exp.yaml --model resnet50.onnx --output results/

# Transformer
ie-run    --config exp.yaml --model llama3.gguf --output results/

# 训练
ie-train  --config train.yaml --model model.json --data data/ --output ckpt/

# 校验
ie-hash   --model model.gguf --all > model.manifest
ie-hash   --check model.manifest --model model.gguf
ie-verify --config verify.yaml --reference trace_ref.json --output report.html
ie-compare --engine results/ --reference vllm_results/ --output diff.html
ie-determinism --config exp.yaml --verify-batch-invariance --seed 42

# 硬件
ie-probe  --backend cuda --output hw_profile.json
ie-probe  --backend cpu --output cpu_profile.json
ie-tune   --config exp.yaml --autotune --output tuning_cache/
ie-report --cross-hardware --input results/ --output cross_hw.html
```


## 14. 借鉴主流框架的优化项

| 优化项 | 借鉴来源 | 适用模型 | 难度 |
|---|---|---|---|
| PagedAttention | vLLM | Transformer | 🟡 中 |
| 连续批处理 | vLLM/TensorRT-LLM | Transformer | 🟡 中 |
| 前缀缓存 | vLLM | Transformer | 🟡 中 |
| KV Cache 量化 | llama.cpp | Transformer | 🟢 低 |
| Chunked Prefill | vLLM | Transformer | 🟡 中 |
| 投机解码 | TensorRT-LLM | Transformer | 🟡 中 |
| 内核自动调优 | TensorRT | 通用 | 🟢 低 |
| 算子融合 | TensorRT | 通用 | 🟢 低 |
| Skip Softmax | TensorRT-LLM | Transformer | 🟡 中 |
| 专家分割+残差预取 | KTransformers/DALI | Transformer (MoE) | 🟡 中 |
| 计算传输重叠 | HiCache | 通用 | 🟡 中 |
| 阶段感知 Arena | llama.cpp | Transformer | 🟡 中 |
| 批次不变性 | vLLM | Transformer | 🟡 中 |
| 张量级哈希 | llama.cpp gguf-hash | 通用 | 🟢 低 |
| 参考追踪 | InferRef | 通用 | 🟡 中 |
| 精度调试管线 | vLLM Neuron | 通用 | 🟡 中 |
| 自适应推理 | TensorRT for RTX | 通用 | 🟡 中 |
| 硬件能力探测 | vLLM platform_utils | 通用 | 🟢 低 |
| **Winograd Conv** | **MNN / oneDNN** | **传统 (CNN)** | **🟡 中** |
| **im2col + GEMM** | **MNN / NCNN** | **传统 (CNN)** | **🟢 低** |
| **INT8 PTQ** | **TensorRT / MNN** | **传统** | **🟡 中** |
| **层融合 (Conv+BN+ReLU)** | **TensorRT / MNN** | **传统** | **🟢 低** |


## 15. 目录结构

```text
infer-engine/
  core/
    base/          Status, Result, Error, Types
    ir/            Node, Graph, OpRegistry, Pass, ControlFlow
    runtime/       Executor, Session, Workspace, Stream
    scheduler/     Scheduler, BackendScheduler, HybridPlanner
    memory/        Allocator, Arena, Pool, TransferManager
    backend/       Backend, Device, BackendRegistry, DeviceSelector
    plugin/        PluginRegistry, PluginMeta, PluginLoader
    hardware/      HardwareProfile, CapabilitySet, HardwareDetector
                   CapabilityNormalizer, OptimizationPlanner
    tuning/        KernelVariant, KernelVariantRegistry
                   RuntimeAutotuner, HardwareAwareCache
  plugins/         稳定插件
    ops/ backends/ passes/ quant/ schedulers/ kv_cache/ samplers/
    optimizers/ lr_schedulers/ losses/ dataloaders/ hardware_probes/
  experimental/    实验特性
    attention/ moe/ kv_cache/ speculative/
  backends/
    cpu/           CpuBackend, simd/, kernels/
    cuda/          CudaBackend, kernels/
    vulkan/        VulkanBackend, shaders/, pipeline/
  ops/
    common/        Add, MatMul, Relu, Softmax, Concat, Slice...
    traditional/   Conv2D, Pool, BN, LSTM, GRU, GBDT...
    transformer/   Attention, RoPE, RMSNorm, Embedding, MoE...
    control/       If, Loop, While, Subgraph
    rule/          RuleMatch, RuleFire, FactAssert
  rules/
    fact/ rule/ agenda/ rete/
  models/
    traditional/   ONNX/TFLite/Caffe 前端
    transformer/   SafeTensors/GGUF 前端
  llm/
    tokenizer/ model_loader/ transformer/ sampler/ streaming/
  hybrid/
    planner/ scheduler/ memory/ orchestrator/
  train/           可选加载
    autograd/ optimizer/ scheduler/ loss/ data/ trainer/ amp/ checkpoint/ distributed/
  validation/      可选加载
    reference_tracer/ tensor_hash/ determinism/ comparator/ cross_framework/ report/
  converter/       模型转换
    format_loader/ op_mapping/ validation/ quantization/
  security/
    tenant/ sandbox/ model_security/ audit/
  fault_tolerance/
    health/ resilience/ debug/
  sdk/
    python/ js/ rust/
  api/
    c_api/ openai_compat/ grpc/ cli/
  observability/
    metrics/ logging/ tracing/ profiler/
  tools/
    trace_exporter/ graph_exporter/ memory_visualizer/
    rule_trace/ llm_visualizer/ training_visualizer/
    validation_visualizer/ hardware_visualizer/ experiment/ report/ web/
  packaging/
    debian/ rpm/ homebrew/ vcpkg/ conda/ pypi/ docker/ wasm/
  benchmarks/
    traditional/ transformer/ rules/ hybrid/ end2end/ compare/
    training/ validation/ profiling/ hardware/
  tests/
    interface/ impl/
  docs/
```


## 16. 分期路线图

| 阶段 | 目标 | 接口 | 实现 | 验收 |
|---|---|---|---|---|
| **P0-MVP** | 跑通静态图 CPU 推理 | 核心接口冻结 | AddOp + CpuBackend + StaticScheduler | `(a+b)*3` 正确 |
| **P0.5** | 接口补全 + 硬件画像 + 双模型接口预留 | 全部接口完整 | 空实现 + 默认方法 | 接口契约测试通过 |
| **P1** | CPU 后端完善 + 硬件探测 | 不变 | 能力位探测 + AVX2/AVX-512 kernel | 小 MLP 对齐 PyTorch |
| **P2** | ONNX 加载器 + Conv/Pool/BN 算子 | 不变 | 传统模型推理链路 | ResNet50 对齐 ONNX Runtime |
| **P3** | GPU 后端（CUDA 或 Vulkan）+ GPU 硬件探测 | 不变 | 能力位驱动 kernel 选择 | 同图 CPU/GPU 一致 |
| **P4** | 后端调度器 + OptimizationPlanner | 不变 | 能力驱动放置、传输 | CPU+GPU 自动调度 |
| **P5** | 规则引擎 | 不变 | RuleEngine + 推理链可视化 | 输出推理链 |
| **P6** | Transformer 算子 + LLM 推理 | 不变 | Attention + RingKVCache | 小模型流式生成 |
| **P7** | PagedAttention + RuntimeAutotuner | 不变 | PagedKVCache 替换 | 显存利用率提升 |
| **P8** | 连续批处理 + HardwareAwareCache | 不变 | ContinuousBatchScheduler | 吞吐提升 |
| **P9** | 混合 H1：层间分割 | 不变 | HybridPlanner | CPU+GPU 串行正确 |
| **P10** | 混合 H2：异步调度 | 不变 | Event 同步 | 设备并行 |
| **P11** | 混合 H3：专家分割 | 不变 | 残差预取 + 负载均衡 | GPU 利用率 > 60% |
| **P12** | 混合 H4：KV 分层 | 不变 | HybridKVCache | 长上下文不 OOM |
| **P13** | 混合 H5：投机解码 | 不变 | SpeculativeDecoder | 加速 1.5x+ |
| **P14** | 混合编排 + INT8 PTQ | 不变 | HybridOrchestrator + INT8 量化 | 传统 + Transformer 混合链路 |
| **P15** | 优化 + 跨硬件对比 | 不变 | 对比报告、消融实验 | 基准报告 |
| **P16** | 发布 | 不变 | 文档、CI、C API、Web Dashboard | 可复现 |
| **P17** | 前沿接入 | 不变 | 插件生态、实验区 | 新插件可独立加载 |
| **P18** | 对比框架 | 不变 | 配置矩阵、报告生成 | 一键跑矩阵 |

**训练路线（T1-T12）** 和**校验路线（V1-V7）** 与推理路线并行推进。


## 17. MVP 代码骨架

```cpp
// main.cpp
int main() {
    // 1. 硬件探测
    auto profile = HardwareDetector::detect_cpu();
    std::cout << "CPU fingerprint: " << profile.fingerprint << "\n";
    std::cout << "AVX2: " << profile.matrix.has_avx2 << "\n";
    std::cout << "AVX-512 VNNI: " << profile.matrix.has_avx512_vnni << "\n";

    // 2. 注册后端
    BackendRegistry::instance().register_backend(
        BackendKind::CPU,
        [] { return std::make_unique<CpuBackend>(); });

    // 3. 构建图
    GraphBuilder builder;
    auto graph = builder.add(2).mul(3).build();

    // 4. 创建引擎
    EngineConfig config;
    config.gpu_backend = GpuBackend::NONE;
    config.default_model_kind = ModelKind::UNKNOWN;
    Engine engine(config);
    engine.load(graph);

    // 5. 准备输入
    Tensor a = Tensor::from_data({1.0f, 2.0f, 3.0f});
    Tensor b = Tensor::from_data({4.0f, 5.0f, 6.0f});

    // 6. 执行
    auto result = engine.infer({{"a", a}, {"b", b}});

    // 7. 验证
    assert(result[0].data<float>()[0] == 15.0f);
}
```

**MVP 模块清单**：

| 模块 | MVP 实现 | 接口完整性 |
|---|---|---|
| Status/Result | `std::expected` | 完整 |
| Tensor/Shape | 基础实现 | 完整 |
| Graph | 静态图 + 控制流字段 | 完整 |
| Op | 通用 + 传统 + Transformer 算子类型 | 完整 |
| OpRegistry | Add/Mul/MatMul | 完整 |
| Backend | CpuBackend 同步 | 完整 |
| BackendRegistry | 静态注册 | 完整 |
| DeviceSelector | 探测返回 NONE | 完整 |
| HardwareDetector | CPU 能力位探测 | 完整 |
| HardwareProfile | 能力位 + 性能参数 | 完整 |
| OptimizationPlanner | 能力驱动决策 | 完整 |
| Allocator | `malloc` 包装 | 完整 |
| KVCachePlugin | 不实现 | 接口已定义 |
| Scheduler | StaticScheduler | 完整 |
| RuleEngine | 不实现 | 接口已定义 |
| TraditionalEngine | 不实现 | 接口已定义 |
| LLMEngine | 不实现 | 接口已定义 |
| PluginRegistry | 空框架 | 完整 |
| RNG | SamplingRNG | 完整 |
| TraceExporter | 不实现 | 接口已定义 |
| GraphExporter | DOT 输出 | 完整 |
| TransferManager | 空框架 | 完整 |
| AutogradEngine | 不实现 | 接口已定义 |
| Trainer | 不实现 | 接口已定义 |
| CheckpointIO | weights_only | 完整 |
| TensorHasher | xxh64 基础 | 完整 |
| Comparator | 基础对比 | 完整 |
| DeterminismGuard | 不实现 | 接口已定义 |
| ModelConverter | 不实现 | 接口已定义 |

**MVP 代码量**：~7000 行。**接口 100% 定义，实现 ~15%**。


## 18. 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| **双模型支持** | **统一 IR + 执行器，分离调度 + 内存** | **共享 90% 核心，差异在 10%** |
| **优化决策依据** | **能力位 + 性能参数** | **不依赖架构名，支持跨代际、跨厂商** |
| **硬件画像** | **CPU 和 GPU 统一框架** | **向量指令集和矩阵单元统一表达** |
| **Kernel 变体绑定** | **能力前置条件 + 模型类别** | **不绑定架构名** |
| **指纹生成** | **能力位 + 性能量级 + 驱动** | **避免架构名带来的误判** |
| **跨厂商兼容** | **能力位归一化** | **NVIDIA TMA ↔ AMD bulk ↔ Intel 2D block** |
| **控制流** | **IR 支持 If/Loop/While** | **传统模型 RNN 需要** |
| **动态维度** | **IR 支持 DynamicDim** | **Transformer batch/seq_len 需要** |
| 接口策略 | 面向未来定义，MVP 填充 | 避免未来重构 |
| GPU 后端 | CUDA 或 Vulkan 二选一 | 简化设备拓扑 |
| 跨设备传输 | CPU↔GPU 两条路径 | 实现量可控 |
| 混合推理 | 两设备协同（CPU + GPU） | 复杂度可控 |
| 训练与推理关系 | 共享内核，分离上层 | 避免推理引入训练开销 |
| Autograd | 可选引擎 | 推理不创建实例 |
| Tensor | 增加 requires_grad 字段 | 默认 false，零开销 |
| Op | backward 默认 Unsupported | 推理算子不重写 |
| Backend | supports_training 默认 false | 推理后端不感知训练 |
| Checkpoint | 推理只读 weights | 忽略训练字段 |
| RNG | 抽象为接口 | 推理/训练用不同实例 |
| 校验架构 | 四层管线 | 参考 vLLM Neuron |
| 张量哈希 | xxh64 默认，sha256 安全 | 参考 llama.cpp |
| 确定性模式 | 可配置，默认关闭 | 参考 vLLM batch invariance |
| 参考追踪 | Trace IR 解耦 PyTorch/ONNX | 引擎侧无框架依赖 |
| 自适应调优 | 编译期分档 + 启动期选择 + 运行期微调 | 参考 TensorRT for RTX |
| 插件边界 | C ABI + C++ 内部接口 | 稳定 ABI，便于动态加载 |
| 实验区 | 独立目录 + 特性门控 | 隔离不稳定代码 |
| Trace 格式 | Chrome Trace | 生态成熟 |
| 图格式 | JSON + DOT | Web UI 和 Graphviz 双消费 |
| 对比框架 | 自研轻量 + 文件存储 | 学习项目，避免重依赖 |
| 可视化 | 自研 Web UI + 可选 Grafana | 兼顾学习与实用 |
| 可复现 | git commit + 硬件画像 + 输入 hash | 保证实验可追溯 |
| CPU 后端 | 参考实现 + oracle | 正确性基准 |
| 演进策略 | 新特性 = 新实现 + 配置切换 | 不改接口，不改上层 |


## 19. 关键风险

| 风险 | 应对 |
|---|---|
| 能力位探测失败 | 回退到最保守的标量实现 |
| 能力位与性能不匹配 | 运行期实测校准，动态调整 |
| 跨厂商能力位归一化错误 | 保守映射，宁缺毋滥 |
| 指纹过细导致缓存失效 | 性能参数量级化，避免过度细分 |
| Vulkan barrier 复杂 | 初期保守全局 barrier |
| 数值一致性 | CPU 后端作 oracle，逐算子对比 |
| 混合推理负载失衡 | 贪心分配 + 动态调整 |
| 编译时间膨胀 | 后端独立静态库，核心不依赖 |
| 过度抽象 | 先垂直切片，再重构 |
| 过早优化 | naive → profiler → 优化 |
| 插件 ABI 不稳定 | 语义化版本 + 兼容性检查 |
| 可视化工具膨胀 | 先用 Perfetto/Grafana，后自研 |
| 接口冻结过早 | P0.5 前允许调整，之后冻结 |
| Vulkan 驱动兼容性 | 限定 Vulkan 1.2+，启动时探测 |
| 训练污染推理 | 训练字段默认关闭，推理零开销 |
| 训练内存膨胀 | 独立 MemoryKind，按需分配 |
| 反向图构建复杂 | backward 默认 Unsupported |
| 确定性模式性能下降 | 默认关闭，调试时开启 |
| 参考追踪开销 | 仅在 `enable_validation` 时启用 |
| 跨框架对齐复杂 | 先对齐 PyTorch/ONNX，再对齐 vLLM/TensorRT |
| **双模型架构冲突** | **统一 IR，分离调度和内存策略** |
| **控制流执行复杂** | **IR 支持，MVP 不实现，逐步接入** |
| **传统模型算子覆盖不足** | **优先 ONNX 子集，逐步扩展** |


## 20. TODO 调研清单

### 20.1 语言与工程
- [ ] C++ 标准（C++20 vs C++23）
- [ ] 构建系统（CMake + Ninja vs Bazel）
- [ ] 依赖管理（vcpkg / Conan / 手动）
- [ ] 错误处理（`std::expected` vs 异常 vs 错误码）
- [ ] 测试框架（GoogleTest / Catch2 / doctest）
- [ ] 基准框架（Google Benchmark / 自研）

### 20.2 硬件能力探测
- [ ] CPU 能力位探测（cpuid / `__builtin_cpu_supports`）
- [ ] AMX 探测（CPUID leaf 7）
- [ ] ARM SVE/SME 探测
- [ ] CUDA 能力位探测（cudaDeviceProp）
- [ ] Vulkan 能力位探测（VkPhysicalDeviceFeatures2）
- [ ] 跨厂商能力位归一化映射
- [ ] ρ 值微基准测试
- [ ] 硬件指纹生成策略
- [ ] 运行期性能测量（NVML / perf / RAPL）

### 20.3 IR 与编译
- [ ] 是否引入 MLIR
- [ ] IR 序列化（JSON / FlatBuffers / Protobuf）
- [ ] 图优化 pass 列表与顺序
- [ ] 多层 IR 下降规则
- [ ] 动态 shape 支持
- [ ] 反向图构建策略
- [ ] **控制流 IR 设计（If/Loop/While）**
- [ ] **RNN 展开策略（静态展开 vs 动态循环）**

### 20.4 后端
- [ ] CPU：SIMD 库、线程池、NUMA、AMX tile 编程
- [ ] CUDA：CUTLASS、CUDA Graph、cooperative groups、TMA
- [ ] Vulkan：Vulkan-Hpp、SPIR-V 编译路径、barrier 优化
- [ ] GPU 后端自动选择策略
- [ ] 后端插件化（C ABI / 动态库）

### 20.5 自适应调优
- [ ] Kernel 变体粒度
- [ ] Cost model 设计
- [ ] 运行期 autotune 预算控制
- [ ] 缓存失效策略
- [ ] 跨会话缓存格式
- [ ] 多 GPU 场景下的缓存隔离

### 20.6 混合推理
- [ ] 切分策略选择与收益评估
- [ ] 放置算法（贪心 / LP / RL）
- [ ] Cost model
- [ ] 负载均衡
- [ ] 跨设备同步原语
- [ ] 权重预取策略
- [ ] KV Cache 淘汰与预取

### 20.7 传统模型
- [ ] **ONNX 算子覆盖范围**
- [ ] **TFLite 算子覆盖范围**
- [ ] **Conv 算法选择（im2col / Winograd / FFT / direct）**
- [ ] **Conv + BN + ReLU 融合**
- [ ] **LSTM/GRU 静态展开 vs 动态循环**
- [ ] **INT8 PTQ 校准策略**
- [ ] **GBDT 推理实现**

### 20.8 Transformer
- [ ] 模型架构支持列表（Llama / Qwen / MoE）
- [ ] Tokenizer 实现（BPE / sentencepiece）
- [ ] 采样策略
- [ ] 结构化输出
- [ ] Attention 变体（FlashAttention / PagedAttention / MLA）
- [ ] KV Cache 量化

### 20.9 训练
- [ ] Autograd 实现方式
- [ ] 反向算子覆盖范围
- [ ] 优化器状态管理
- [ ] 混合精度训练策略
- [ ] 梯度累积与裁剪
- [ ] 梯度检查点
- [ ] 分布式训练（DP / TP / PP / ZeRO）
- [ ] Checkpoint 格式与兼容性

### 20.10 校验与调试
- [ ] Trace IR 格式设计
- [ ] PyTorch / ONNX 追踪脚本实现
- [ ] 哈希算法选择
- [ ] 对比容差校准策略
- [ ] 批次不变性的 Kernel 实现
- [ ] vLLM / TensorRT-LLM / ONNX Runtime 输出格式适配
- [ ] HTML 报告生成

### 20.11 模型转换
- [ ] ONNX / TFLite / SafeTensors / GGUF 加载器
- [ ] 五阶段转换验证管线
- [ ] 算子映射注册表
- [ ] GGUF / INT8 量化工具链

### 20.12 内存
- [ ] 内存池策略
- [ ] 跨设备传输路径
- [ ] Pinned memory 池大小
- [ ] OOM 降级路径
- [ ] 激活内存管理
- [ ] 梯度内存复用
- [ ] **权重常驻内存策略**

### 20.13 调度
- [ ] 调度算法
- [ ] 动态批 / 连续批
- [ ] 优先级与抢占
- [ ] 会话管理
- [ ] **传统模型静态 batch 策略**

### 20.14 可演进
- [ ] 插件 ABI 边界
- [ ] 动态库加载方案
- [ ] API 版本兼容
- [ ] 特性门控机制
- [ ] 实验区隔离

### 20.15 可视化
- [ ] Trace 格式（Chrome / Perfetto / OTLP）
- [ ] 图可视化库
- [ ] 前端框架
- [ ] 时间线 UI
- [ ] 内存可视化
- [ ] 规则推理链可视化
- [ ] Attention 热力图
- [ ] Dashboard
- [ ] 训练曲线可视化
- [ ] 校验报告可视化
- [ ] 硬件画像可视化
- [ ] 调优决策可视化
- [ ] **传统模型 Conv 层可视化**

### 20.16 对比框架
- [ ] 实验元数据存储
- [ ] 配置矩阵定义
- [ ] 结果统计（多次运行、方差、置信区间）
- [ ] 报告生成
- [ ] 可复现性保障
- [ ] CI 自动跑基准
- [ ] 跨硬件对比报告

### 20.17 前沿方向优先级
- [ ] 量化：FP8 / MXFP4 / AWQ / GPTQ / GGUF / INT8 PTQ
- [ ] 注意力：FlashAttention-2/3 / PagedAttention / MLA
- [ ] MoE：DeepSeek MoE / Mixtral
- [ ] KV Cache：H2O / StreamingLLM / KV 量化
- [ ] 投机解码：Medusa / EAGLE / Lookahead
- [ ] 新架构：Mamba / RWKV / RetNet
- [ ] 并行：TP / PP / EP
- [ ] 结构化输出：JSON Schema / FSM
- [ ] 多模态：ViT / CLIP / Whisper
- [ ] 工具调用：Function Calling / MCP
- [ ] 训练优化：LoRA / QLoRA / ZeRO / FSDP
- [ ] **传统模型：Winograd / FFT Conv / INT8 PTQ**

### 20.18 部署与安全
- [ ] API 框架
- [ ] C API 设计
- [ ] Python binding
- [ ] 容器化
- [ ] 沙箱与权限
- [ ] 多租户隔离
- [ ] 审计日志
- [ ] 模型许可证


## 21. 总结

**架构主线**：

```text
核心：IR + Graph + Backend 接口 + 内存抽象 + 调度骨架
硬件：HardwareProfile（能力位 + 性能参数）→ OptimizationPlanner → 决策
后端：CPU（参考） + 单一 GPU（CUDA 或 Vulkan）
双模型：
  传统模型：ONNX/TFLite 前端 + Conv/Pool/BN/LSTM 算子 + 静态调度 + INT8 PTQ
  Transformer：SafeTensors/GGUF 前端 + Attention/RoPE 算子 + 迭代调度 + KV Cache
  统一：IR + 执行器 + 硬件画像 + 插件框架 + 可视化
混合：HybridPlanner + HybridScheduler + HybridMemoryManager（两设备）
范式：规则、传统、Transformer、混合编排，统一图表示
训练：AutogradEngine + Trainer + Optimizer（可选加载，推理零开销）
校验：TraceIR + TensorHasher + DeterminismGuard + Comparator（可选加载）
转换：ModelConverter + 五阶段验证（可选加载）
演进：插件注册表 + 实验区 + 特性门控
观测：Trace + Graph + Metrics 三类数据流
对比：配置矩阵 + 实验仓库 + 报告（含跨硬件对比）
```

**双模型支持的核心策略**：

> **统一 IR 和执行器，分离调度策略和内存策略。** 传统模型和 Transformer 共享 90% 的核心（IR、Graph、Backend、Allocator、OpRegistry、HardwareProfile、PluginRegistry），差异集中在 10% 的调度（静态 vs 迭代）、内存（权重常驻 vs KV Cache）、算子（Conv/LSTM vs Attention/RoPE）上。

**传统模型支持**：
- ONNX/TFLite/Caffe 前端
- Conv/Pool/BN/LSTM/GRU/GBDT 算子
- im2col + GEMM / Winograd / FFT Conv 算法
- Conv + BN + ReLU 层融合
- INT8 PTQ 校准
- 静态调度器，固定 batch
- 权重常驻内存

**Transformer 支持**：
- SafeTensors/GGUF 前端
- Attention/RoPE/RMSNorm/Embedding/MoE 算子
- FlashAttention / PagedAttention
- KV Cache 管理
- FP8 / KV 量化
- 迭代调度器，连续批处理
- 前缀缓存 / 投机解码

**核心修正**：
> **优化决策只依赖硬件能力画像（能力位 + 性能参数），不依赖架构名、型号名或代际名。** CPU 的 AVX2/AVX-512/VNNI/AMX 差异和 GPU 的 RTX 3050/H200 差异，统一用能力位表达。

**推理零开销保证**：
- 推理张量 `requires_grad = false`
- 推理图 `is_training = false`
- 推理后端 `supports_training = false`
- 推理不创建 `AutogradEngine`
- 推理不加载训练、校验、转换插件
- 推理加载只读 `model_weights`
- 校验仅在 `enable_validation = true` 时启用

**推进策略**：
1. **P0-MVP**（~7000 行）：跑通静态图 CPU 推理 + 硬件能力探测。
2. **P0.5**：冻结全部接口（含双模型、训练、校验、转换预留）。
3. **P1-P18**：推理演进，训练/校验/转换接口不变。
4. **V1-V7**：校验工具逐步接入。
5. **T1-T12**：推理内核稳定后，逐步实现训练模块。

**一句话总结**：

> **统一 IR 和执行器支持传统模型与 Transformer；优化决策只依赖硬件能力画像（能力位 + 性能参数）；传统模型走 ONNX/TFLite + Conv/LSTM + 静态调度 + INT8 PTQ，Transformer 走 SafeTensors/GGUF + Attention/RoPE + 迭代调度 + KV Cache；两者共享 90% 核心，差异在 10% 的调度、内存和算子实现上。所有前沿方法都是插件，所有实验都可对比，所有过程都可观测。**