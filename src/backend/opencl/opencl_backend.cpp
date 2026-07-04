#include "compute_engine/backend/opencl_backend.h"
#include "compute_engine/platform/logger.h"

namespace compute_engine {

// ─── OpenCLDevice ───

std::string OpenCLDevice::name() const {
    return OpenCLContext::Instance().device_name;
}

std::string OpenCLDevice::version() const {
    return OpenCLContext::Instance().device_version;
}

Status OpenCLDevice::Initialize() {
    return OpenCLContext::Instance().Init();
}

Status OpenCLDevice::Finalize() {
    OpenCLContext::Instance().Destroy();
    return Status::OK();
}

bool OpenCLDevice::IsAvailable() const {
    cl_uint count = 0;
    return clGetPlatformIDs(0, nullptr, &count) == CL_SUCCESS && count > 0;
}

std::vector<DeviceInfo> OpenCLDevice::EnumDevices() const {
    return OpenCLContext::Instance().EnumerateDevices();
}

size_t OpenCLDevice::GetMemoryUsage() const {
    return 0;
}

size_t OpenCLDevice::GetMaxMemory() const {
    return static_cast<size_t>(OpenCLContext::Instance().total_global_mem);
}

// ─── OpenCLStreamManager ───

OpenCLStreamManager::OpenCLStreamManager() {}
OpenCLStreamManager::~OpenCLStreamManager() {
    SynchronizeAll();
}

int OpenCLStreamManager::CreateStream() {
    OpenCLContext& ctx = OpenCLContext::Instance();
    std::lock_guard<std::mutex> lock(mutex_);

    cl_int err = 0;
    cl_queue_properties props[] = {0};
    cl_command_queue q = clCreateCommandQueueWithProperties(
        ctx.context, ctx.device, props, &err);
    if (!q) return -1;

    int id = static_cast<int>(queues_.size());
    queues_.push_back(q);
    LOGI("OpenCL stream created: id={}", id);
    return id;
}

Status OpenCLStreamManager::DestroyStream(int stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_id >= 0 && stream_id < static_cast<int>(queues_.size())) {
        clReleaseCommandQueue(queues_[stream_id]);
        queues_[stream_id] = nullptr;
    }
    return Status::OK();
}

Status OpenCLStreamManager::SynchronizeStream(int stream_id) {
    OpenCLContext& ctx = OpenCLContext::Instance();
    cl_command_queue q = (stream_id >= 0 && stream_id < static_cast<int>(queues_.size()))
                         ? queues_[stream_id] : ctx.command_queue;
    clFinish(q);
    return Status::OK();
}

Status OpenCLStreamManager::SynchronizeAll() {
    clFinish(OpenCLContext::Instance().command_queue);
    return Status::OK();
}

Status OpenCLStreamManager::RecordEvent(int stream_id, int event_id) {
    CE_UNUSED(stream_id);
    CE_UNUSED(event_id);
    return Status::OK();
}

Status OpenCLStreamManager::WaitEvent(int stream_id, int event_id) {
    CE_UNUSED(stream_id);
    CE_UNUSED(event_id);
    return Status::OK();
}

Status OpenCLStreamManager::QueryEvent(int event_id, bool* completed) {
    if (completed) *completed = true;
    CE_UNUSED(event_id);
    return Status::OK();
}

// ─── OpenCLBackend ───

OpenCLBackend::OpenCLBackend() {
    LOGI("OpenCLBackend created");
}

OpenCLBackend::~OpenCLBackend() {
    LOGI("OpenCLBackend destroyed");
}

} // namespace compute_engine
