#include <gtest/gtest.h>

#include <cstdint>

#include "backends/cpu/cpu_backend.h"
#include "core/backend/backend_matrix.h"
#include "core/backend/backend_registry.h"
#include "core/backend/device_selector.h"
#include "core/backend/engine_config.h"

using namespace ie;

TEST(BackendRegistryTest, RegistersAndCreates) {
    auto& reg = BackendRegistry::instance();
    auto st = reg.register_backend(BackendKind::CUDA,
                                   [] { return std::unique_ptr<Backend>{}; });
    EXPECT_TRUE(st.has_value() || st.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(reg.contains(BackendKind::CUDA));
    auto created = reg.create(BackendKind::CUDA);
    ASSERT_FALSE(created.has_value());
    EXPECT_EQ(created.error().code, Error::Code::Runtime);
}

TEST(BackendRegistryTest, DuplicateRegistrationIsRejected) {
    auto& reg = BackendRegistry::instance();
    ASSERT_TRUE(reg.register_backend(BackendKind::VULKAN,
                                     [] { return std::unique_ptr<Backend>{}; })
                    .has_value());
    auto st = reg.register_backend(BackendKind::VULKAN,
                                   [] { return std::unique_ptr<Backend>{}; });
    EXPECT_EQ(st.error().code, Error::Code::AlreadyExists);
}

TEST(BackendRegistryTest, EmptyFactoryIsRejected) {
    auto& reg = BackendRegistry::instance();
    auto st = reg.register_backend(BackendKind::CPU, BackendRegistry::Factory{});
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().code, Error::Code::InvalidArgument);
}

TEST(DeviceSelectorTest, ReturnsNoneForP0) {
    EXPECT_EQ(DeviceSelector::select(), GpuBackend::NONE);
}

TEST(EngineConfigTest, RejectsEmptyDeviceSet) {
    EngineConfig cfg;
    cfg.enable_cpu = false;
    cfg.gpu_backend = GpuBackend::NONE;
    auto st = validate(cfg);
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().code, Error::Code::InvalidArgument);
}

TEST(BackendMatrixTest, DefaultsToUnsupported) {
    BackendMatrix m;
    EXPECT_FALSE(m.supports(BackendKind::CPU, OpType::Add));
}

TEST(CpuBackendTest, InitDetectsProfileAndIsIdempotent) {
    CpuBackend backend;
    ASSERT_TRUE(backend.init().has_value());
    ASSERT_TRUE(backend.init().has_value());
    auto profile = backend.hardware_profile();
    ASSERT_TRUE(profile.has_value());
    EXPECT_GT(profile->cu.num_cores, 0);
    auto caps = backend.caps();
    EXPECT_TRUE(caps.supports_deterministic_kernels);
    EXPECT_TRUE(caps.supports_batch_invariant);
    ASSERT_TRUE(backend.shutdown().has_value());
    ASSERT_TRUE(backend.shutdown().has_value());
}

TEST(CpuBackendTest, AllocIsAlignedAndFreeIsSafe) {
    CpuBackend backend;
    ASSERT_TRUE(backend.init().has_value());
    auto ptr = backend.alloc(65);
    ASSERT_TRUE(ptr.has_value());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(*ptr) % 64, 0u);
    backend.free(*ptr);
    backend.free(*ptr);
    backend.free(nullptr);
}
