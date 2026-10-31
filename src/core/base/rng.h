#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>

#include "core/base/error.h"
#include "core/memory/allocator.h"

namespace ie {

class RNG {
public:
    virtual ~RNG() = default;

    virtual std::uint64_t next_u64() = 0;
    virtual float next_float() = 0;
    virtual void set_seed(std::uint64_t seed) = 0;
    virtual Status save_state(BufferHandle state) = 0;
    virtual Status load_state(BufferHandle state) = 0;
};

class SamplingRNG final : public RNG {
public:
    SamplingRNG() : engine_(0) {}

    std::uint64_t next_u64() override { return engine_(); }

    float next_float() override {
        return std::uniform_real_distribution<float>(0.0f, 1.0f)(engine_);
    }

    void set_seed(std::uint64_t seed) override { engine_.seed(seed); }

    Status save_state(BufferHandle /*state*/) override {
        return Error::unsupported("rng state serialization not implemented in P0");
    }

    Status load_state(BufferHandle /*state*/) override {
        return Error::unsupported("rng state serialization not implemented in P0");
    }

private:
    std::mt19937_64 engine_;
};

class RNGRegistry {
public:
    static RNGRegistry& instance() {
        static RNGRegistry registry;
        return registry;
    }

    void register_rng(const std::string& name, RNG* rng) { rngs_[name] = rng; }

    RNG* get(const std::string& name) {
        auto it = rngs_.find(name);
        return it == rngs_.end() ? nullptr : it->second;
    }

private:
    RNGRegistry() = default;
    std::unordered_map<std::string, RNG*> rngs_;
};

}  // namespace ie
