#include "compute_engine/backend/cpu_backend.h"
#include <cstdlib>
#include <cstring>
#include "compute_engine/platform/logger.h"

namespace compute_engine {

Status CPUDevice::Initialize() {
    LOGI("CPU device initialized");
    return Status::OK();
}
Status CPUDevice::Finalize() {
    LOGI("CPU device finalized");
    return Status::OK();
}

std::vector<DeviceInfo> CPUDevice::EnumDevices() const {
    DeviceInfo info;
    info.name = "CPU";
    info.version = "1.0";
    info.type = DeviceType::CPU;
    LOGI("CPU device enumerated");
    return {info};
}

size_t CPUDevice::GetMemoryUsage() const { return 0; }
size_t CPUDevice::GetMaxMemory() const { return static_cast<size_t>(-1); }

int CPUStreamManager::CreateStream() {
    LOGI("CPU stream created");
    return 0;
}
Status CPUStreamManager::DestroyStream(int) { return Status::OK(); }
Status CPUStreamManager::SynchronizeStream(int) { return Status::OK(); }
Status CPUStreamManager::SynchronizeAll() { return Status::OK(); }
Status CPUStreamManager::RecordEvent(int, int) { return Status::OK(); }
Status CPUStreamManager::WaitEvent(int, int) { return Status::OK(); }
Status CPUStreamManager::QueryEvent(int, bool* completed) {
    if (completed) *completed = true;
    return Status::OK();
}

Status CPUKernelExecutor::CompileKernel(
    const std::string&, const std::vector<uint8_t>&, KernelHandle* handle) {
    if (handle) *handle = 1;
    LOGI("CPU kernel compiled (stub)");
    return Status::OK();
}

Status CPUKernelExecutor::LaunchKernel(
    KernelHandle, const std::vector<BufferArg>&, const LaunchConfig&, int) {
    LOGI("CPU kernel launched (stub)");
    return Status::OK();
}

Status CPUKernelExecutor::ReleaseKernel(KernelHandle) { return Status::OK(); }
Status CPUKernelExecutor::BeginGraphCapture(int) { return Status::NotImplemented("CPU does not support graph capture"); }
Status CPUKernelExecutor::EndGraphCapture(GraphHandle*) { return Status::NotImplemented("CPU does not support graph capture"); }
Status CPUKernelExecutor::LaunchGraph(GraphHandle) { return Status::NotImplemented("CPU does not support graph capture"); }

CPUBackend::CPUBackend() {
    LOGI("CPUBackend created");
}
CPUBackend::~CPUBackend() {
    LOGI("CPUBackend destroyed");
}

} // namespace compute_engine
