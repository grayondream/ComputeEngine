#include <gtest/gtest.h>

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
