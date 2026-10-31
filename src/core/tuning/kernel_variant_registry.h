#pragma once

#include <vector>

#include "core/tuning/kernel_variant.h"

namespace ie {

class KernelVariantRegistry {
public:
    void register_variant(KernelVariant variant) {
        variants_.push_back(std::move(variant));
    }

    std::vector<KernelVariant> list(OpType op) const {
        std::vector<KernelVariant> matches;
        for (const KernelVariant& variant : variants_) {
            if (variant.op == op) {
                matches.push_back(variant);
            }
        }
        return matches;
    }

private:
    std::vector<KernelVariant> variants_;
};

}  // namespace ie
