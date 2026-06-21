#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/core/dtype.h"
#include "compute_engine/core/shape.h"
#include "compute_engine/op/op.h"
#include "compute_engine/backend/backend.h"
#include <string>
#include <vector>
#include <functional>
#include <mutex>

namespace compute_engine {

class OpSchema {
public:
    std::string name;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<DType> supported_dtypes = {DType::FLOAT32};

    using ShapeInferenceFn = std::function<Status(
        const std::vector<Shape>& inputs,
        std::vector<Shape>& outputs)>;

    ShapeInferenceFn shape_inference;

    using ConstFoldFn = std::function<Status(
        const std::vector<TensorPtr>& inputs,
        std::vector<TensorPtr>& outputs)>;

    ConstFoldFn const_fold;
};

class OpSchemaRegistry {
public:
    static OpSchemaRegistry& Instance();

    void Register(const std::string& op_type, OpSchema schema);
    const OpSchema* Get(const std::string& op_type) const;
    bool Has(const std::string& op_type) const;
    std::vector<std::string> ListOps() const;

private:
    std::unordered_map<std::string, OpSchema> schemas_;
};

struct KernelCacheKey {
    std::string op_type;
    DeviceType device = DeviceType::CPU;
    DType dtype = DType::FLOAT32;
    uint64_t shape_hash = 0;
    uint64_t compile_options = 0;

    bool operator==(const KernelCacheKey& o) const {
        return op_type == o.op_type && device == o.device
            && dtype == o.dtype && shape_hash == o.shape_hash
            && compile_options == o.compile_options;
    }
};

} // namespace compute_engine

namespace std {
template<>
struct hash<compute_engine::KernelCacheKey> {
    size_t operator()(const compute_engine::KernelCacheKey& k) const {
        size_t h = hash<string>()(k.op_type);
        h ^= hash<uint8_t>()(static_cast<uint8_t>(k.device)) << 8;
        h ^= hash<uint8_t>()(static_cast<uint8_t>(k.dtype)) << 16;
        h ^= hash<uint64_t>()(k.shape_hash) << 24;
        h ^= hash<uint64_t>()(k.compile_options) << 48;
        return h;
    }
};
} // namespace std

namespace compute_engine {

class KernelRegistry {
public:
    using KernelFactory = std::function<Status(
        KernelExecutor* exec, KernelHandle* handle)>;

    static KernelRegistry& Instance();

    void Register(
        const std::string& op_type,
        DeviceType device,
        int priority,
        KernelFactory factory);

    Status GetOrCompile(
        const std::string& op_type,
        KernelExecutor* exec,
        const KernelCacheKey& key,
        KernelHandle* handle);

    bool Has(const std::string& op_type, DeviceType device) const;
    void ClearCache();

private:
    struct Entry {
        DeviceType device;
        int priority;
        KernelFactory factory;
    };
    std::vector<std::pair<std::string, std::vector<Entry>>> entries_;
    std::vector<std::pair<KernelCacheKey, KernelHandle>> cache_;
};

#define REGISTER_OP_SCHEMA(name, schema)                              \
    static struct AutoSchema_##name {                                 \
        AutoSchema_##name() {                                         \
            OpSchemaRegistry::Instance().Register(#name, schema);     \
        }                                                             \
    } s_autoschema_##name

#define REGISTER_KERNEL(op_type, device, priority, factory)           \
    static struct AutoKernel_##op_type##_##device {                   \
        AutoKernel_##op_type##_##device() {                           \
            KernelRegistry::Instance().Register(                      \
                #op_type, device, priority, factory);                 \
        }                                                             \
    } s_autokernel_##op_type##_##device

} // namespace compute_engine
