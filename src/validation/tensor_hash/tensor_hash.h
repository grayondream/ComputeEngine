#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/memory/tensor.h"

namespace ie {

// Forward declarations keep the return types usable inside the class without
// requiring the structs (which reference TensorHasher::HashType) to be defined
// first. The structs are defined after the class.
struct HashManifest;
struct VerifyResult;

class TensorHasher {
public:
    enum class HashType { XXH64, SHA1, SHA256, UUID };

    static Result<std::string> hash_tensor(
        const Tensor& tensor, HashType hash_type = HashType::XXH64);
    static Result<HashManifest> hash_model(const Graph& graph,
                                           HashType hash_type);
    static Result<VerifyResult> verify_manifest(
        const Graph& graph, const HashManifest& manifest);
};

struct HashManifest {
    std::string model_hash;
    std::unordered_map<std::string, std::string> tensor_hashes;
    TensorHasher::HashType hash_type{TensorHasher::HashType::XXH64};
};

struct VerifyResult {
    bool passed{false};
    std::vector<std::string> mismatched_tensors;
    std::string first_mismatch;
};

inline Result<std::string> TensorHasher::hash_tensor(const Tensor& tensor,
                                                     HashType hash_type) {
    (void)tensor;
    (void)hash_type;
    return Error::unsupported("tensor hasher not implemented in P0.5");
}

inline Result<HashManifest> TensorHasher::hash_model(const Graph& graph,
                                                     HashType hash_type) {
    (void)graph;
    (void)hash_type;
    return Error::unsupported("tensor hasher not implemented in P0.5");
}

inline Result<VerifyResult> TensorHasher::verify_manifest(
    const Graph& graph, const HashManifest& manifest) {
    (void)graph;
    (void)manifest;
    return Error::unsupported("tensor hasher not implemented in P0.5");
}

}  // namespace ie
