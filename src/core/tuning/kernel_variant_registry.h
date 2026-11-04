#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/base/error.h"
#include "core/base/shape.h"
#include "core/hardware/profile.h"
#include "core/ir/model_kind.h"
#include "core/ir/op_type.h"
#include "core/tuning/kernel_variant.h"

namespace ie {

class KernelVariantRegistry {
public:
    void register_variant(OpType op, DType dtype, KernelVariant variant) {
        entries_.push_back(Entry{op, dtype, std::move(variant)});
    }

    // Returns every registered variant for (op, dtype) whose requirements are
    // satisfied by the profile and model kind. An empty match is not an error.
    Result<std::vector<KernelVariant>> select(
        OpType op, DType dtype, const Shape& /*shape*/,
        const HardwareProfile& profile,
        ModelKind kind = ModelKind::UNKNOWN) const {
        std::vector<KernelVariant> matches;
        for (const Entry& entry : entries_) {
            if (entry.op != op || entry.dtype != dtype) {
                continue;
            }
            if (!matches_profile(entry.variant, profile, kind)) {
                continue;
            }
            matches.push_back(entry.variant);
        }
        return matches;
    }

private:
    struct Entry {
        OpType op;
        DType dtype;
        KernelVariant variant;
    };

    static bool matches_profile(const KernelVariant& variant,
                                const HardwareProfile& profile,
                                ModelKind kind) {
        if (variant.target_kind != ModelKind::UNKNOWN &&
            variant.target_kind != kind) {
            return false;
        }
        for (const std::string& cap : variant.req.required_caps) {
            if (!profile.caps.has(cap)) {
                return false;
            }
        }
        for (const std::string& cap : variant.req.forbidden_caps) {
            if (profile.caps.has(cap)) {
                return false;
            }
        }
        const float ratio = profile.matrix.tensor_to_vector_ratio;
        if (ratio < variant.req.min_tensor_to_vector_ratio ||
            ratio > variant.req.max_tensor_to_vector_ratio) {
            return false;
        }
        if (profile.mem.vram_total < variant.req.min_vram) {
            return false;
        }
        if (profile.mem.bandwidth_gbps < variant.req.min_bandwidth) {
            return false;
        }
        if (profile.cu.l3_cache_size < variant.req.min_l3_cache) {
            return false;
        }
        return true;
    }

    std::vector<Entry> entries_;
};

}  // namespace ie
