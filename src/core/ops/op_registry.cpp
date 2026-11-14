#include "core/ops/op_registry.h"

#include "core/ops/common/add_op.h"
#include "core/ops/common/matmul_op.h"
#include "core/ops/common/mul_op.h"
#include "core/ops/common/relu_op.h"
#include "core/ops/common/softmax_op.h"
#include "ops/traditional/activation_ops.h"
#include "ops/traditional/batch_norm_op.h"
#include "ops/traditional/conv2d_op.h"
#include "ops/traditional/data_ops.h"
#include "ops/traditional/gemm_op.h"
#include "ops/traditional/layout_ops.h"
#include "ops/traditional/pool_ops.h"

namespace ie {

OpRegistry& OpRegistry::instance() {
    static OpRegistry registry;
    return registry;
}

Status OpRegistry::register_op(OpType op, Creator creator) {
    if (!creator) {
        return Error::invalid_argument("empty op creator");
    }
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
    auto instance = creator();
    if (!instance) {
        return Error::runtime("op factory returned null");
    }
    return instance;
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
    ignore_exists(registry.register_op(
        OpType::Relu, [] { return std::make_unique<ReluOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Softmax, [] { return std::make_unique<SoftmaxOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Conv2D, [] { return std::make_unique<Conv2DOp>(); }));
    ignore_exists(registry.register_op(
        OpType::DepthwiseConv2D,
        [] { return std::make_unique<DepthwiseConv2DOp>(); }));
    ignore_exists(registry.register_op(
        OpType::MaxPool2D, [] { return std::make_unique<MaxPool2DOp>(); }));
    ignore_exists(registry.register_op(
        OpType::AvgPool2D, [] { return std::make_unique<AvgPool2DOp>(); }));
    ignore_exists(registry.register_op(
        OpType::GlobalAvgPool,
        [] { return std::make_unique<GlobalAvgPoolOp>(); }));
    ignore_exists(registry.register_op(
        OpType::BatchNorm, [] { return std::make_unique<BatchNormOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Gemm, [] { return std::make_unique<GemmOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Flatten, [] { return std::make_unique<FlattenOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Reshape, [] { return std::make_unique<ReshapeOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Transpose, [] { return std::make_unique<TransposeOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Concat, [] { return std::make_unique<ConcatOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Slice, [] { return std::make_unique<SliceOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Pad, [] { return std::make_unique<PadOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Upsample, [] { return std::make_unique<UpsampleOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Gather, [] { return std::make_unique<GatherOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Reduce, [] { return std::make_unique<ReduceOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Clip, [] { return std::make_unique<ClipOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Sigmoid, [] { return std::make_unique<SigmoidOp>(); }));
    ignore_exists(registry.register_op(
        OpType::Tanh, [] { return std::make_unique<TanhOp>(); }));
}

}  // namespace ie
