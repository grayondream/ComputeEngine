#pragma once

#include "core/base/error.h"
#include "core/runtime/execution_context.h"
#include "core/scheduler/plan.h"
#include "llm/model_loader/model_loader.h"

namespace ie {

// Transformer model contract (docs/readme.md §8.2). One forward() consumes a
// BatchPlan (prefills + decodes) and writes into the execution context.
class TransformerModel {
public:
    virtual ~TransformerModel() = default;

    virtual Status load(const ModelConfig& config) = 0;
    virtual Status forward(const BatchPlan& plan, ExecutionContext& context) = 0;
};

}  // namespace ie
