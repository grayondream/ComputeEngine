#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/ir/model_kind.h"
#include "core/ir/op_type.h"
#include "core/ops/op.h"

namespace ie {

class OpRegistry {
public:
    using Creator = std::function<std::unique_ptr<Op>()>;

    static OpRegistry& instance();

    Status register_op(OpType op, Creator creator);
    Result<std::unique_ptr<Op>> create(OpType op);
    std::vector<OpType> list_by_kind(ModelKind kind) const;

private:
    OpRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<OpType, Creator> creators_;
};

// Registers Add/Mul/MatMul. Idempotent: already-registered ops are ignored.
void register_builtin_ops(OpRegistry& registry);

}  // namespace ie
