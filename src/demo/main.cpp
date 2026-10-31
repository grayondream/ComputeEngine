#include <cassert>
#include <iostream>
#include <unordered_map>

#include "backends/cpu/cpu_backend.h"
#include "core/backend/backend_registry.h"
#include "core/hardware/detector.h"
#include "core/ir/graph_builder.h"
#include "core/runtime/engine.h"

using namespace ie;

int main() {
    auto profile = HardwareDetector::detect_cpu();
    if (!profile.has_value()) {
        std::cerr << "cpu detect failed: " << profile.error().message << "\n";
        return 1;
    }
    std::cout << "CPU fingerprint: " << profile->fingerprint << "\n";
    std::cout << "AVX2: " << profile->matrix.has_avx2 << "\n";
    std::cout << "AVX-512 VNNI: " << profile->matrix.has_avx512_vnni << "\n";

    (void)BackendRegistry::instance().register_backend(
        BackendKind::CPU, [] { return std::make_unique<CpuBackend>(); });

    GraphBuilder builder;
    auto graph = std::move(builder).add(2).mul(3.0).build();

    EngineConfig config;
    config.gpu_backend = GpuBackend::NONE;
    config.default_model_kind = ModelKind::UNKNOWN;
    Engine engine(config);
    if (!engine.load(graph).has_value()) {
        std::cerr << "load failed\n";
        return 1;
    }

    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({1.0f, 2.0f, 3.0f}));
    inputs.emplace("b", Tensor::from_data({4.0f, 5.0f, 6.0f}));

    auto result = engine.infer(inputs);
    if (!result.has_value()) {
        std::cerr << "infer failed: " << result.error().message << "\n";
        return 1;
    }
    const float* out = result->at(0).data<float>();
    assert(out[0] == 15.0f);
    std::cout << "result: " << out[0] << ", " << out[1] << ", " << out[2]
              << "\n";
    return out[0] == 15.0f && out[1] == 21.0f && out[2] == 27.0f ? 0 : 2;
}
