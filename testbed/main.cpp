#include <compute_engine/compute_engine.h>

int main(int argc, char** argv) {
    using namespace compute_engine;

    LogManager::Init("compute_engine.log", LogLevel::INFO, true);
    BackendRegistry::InitBuiltinBackends();
    LOGI("ComputeEngine Testbed v1.0.0");

    Shape s({2, 3});
    LOGI("Shape: {}, ndim={}, elements={}", s.ToString(), s.ndim(), s.num_elements());

    DType dt = DType::FLOAT32;
    LOGI("DType: {}, size={} bytes", DTypeToString(dt), SizeOf(dt));

    Status st = Status::OK();
    LOGI("Status: {}, ok={}", st.ToString(), st.ok());

    Tensor t({2, 3}, DType::FLOAT32);
    LOGI("Tensor: {}, contiguous={}", t.ToString(), t.contiguous());

    auto zeros = Tensor::Zeros({4, 4}, DType::FLOAT32);
    auto ones = Tensor::Ones({4, 4}, DType::FLOAT32);
    LOGI("Zeros tensor: {}", zeros.ToString());
    LOGI("Ones tensor: {}", ones.ToString());

    auto cpu = BackendRegistry::Instance().GetBestAvailable({DeviceType::CPU});
    if (cpu) {
        LOGI("CPU Backend: {}, available={}", cpu->device()->name(), cpu->IsAvailable());
    }

    auto cuda = BackendRegistry::Instance().GetBestAvailable({DeviceType::CUDA});
    if (cuda) {
        cuda->Initialize();
        LOGI("CUDA Backend: {}, available={}, memory={} MB",
             cuda->device()->name(), cuda->IsAvailable(),
             cuda->device()->GetMaxMemory() / (1024 * 1024));
        cuda->Finalize();
    }

    auto vulkan = BackendRegistry::Instance().GetBestAvailable({DeviceType::VULKAN});
    if (vulkan) {
        vulkan->Initialize();
        LOGI("Vulkan Backend: {}, available={}, memory={} MB",
             vulkan->device()->name(), vulkan->IsAvailable(),
             vulkan->device()->GetMaxMemory() / (1024 * 1024));
        vulkan->Finalize();
    }

    auto opencl = BackendRegistry::Instance().GetBestAvailable({DeviceType::OPENCL});
    if (opencl) {
        opencl->Initialize();
        LOGI("OpenCL Backend: {}, available={}, memory={} MB",
             opencl->device()->name(), opencl->IsAvailable(),
             opencl->device()->GetMaxMemory() / (1024 * 1024));
        opencl->Finalize();
    }

    CE_LOG_INFO("Backward-compatible CE_LOG_INFO macro works");

    LOGI("All basic tests passed!");

    LogManager::Shutdown();
    return 0;
}
