#include "core/ops/op_registry.h"

#include "core/ops/common/add_op.h"
#include "core/ops/common/matmul_op.h"
#include "core/ops/common/mul_op.h"

namespace ie {

OpRegistry& OpRegistry::instance() {
    static OpRegistry registry;
    return registry;
}

Status OpRegistry::register_op(OpType op, Creator creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (creators_.find(op) != creators_.end()) {
        return Error::already_exists("op already registered");
    }
    creators_.emplace(op, std::move(creator));
    return {};
}

Result<std::unique_ptr<Op>> OpRegistry::create(OpType op) {
    Creator creator;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = creators_.find(op);
        if (it == creators_.end()) {
            return Error::not_found("no kernel for op");
        }
        creator = it->second;
    }
    return creator();
}

std::vector<OpType> OpRegistry::list_by_kind(ModelKind /*kind*/) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<OpType> ops;
    ops.reserve(creators_.size());
    for (const auto& [op, _] : creators_) {
        ops.push_back(op);
    }
    return ops;
}

void register_builtin_ops(OpRegistry& registry) {
    const auto ignore_exists = [](Status status) {
        (void)status;
    };
    ignore_exists(registry.register_op(
        OpType::Add, [] { return std::make_unique<AddOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Mul, [] { return std::make_unique<MulOp>(); }));
    ignore_exists(registry.register_op(
        OpType::MatMul, [] { return std::make_unique<MatMulOp>(); }));
}

}  // namespace ie
