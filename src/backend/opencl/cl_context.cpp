#include "compute_engine/backend/opencl_backend.h"
#include "compute_engine/platform/logger.h"
#include <cstring>
#include <vector>

namespace compute_engine {

OpenCLContext& OpenCLContext::Instance() {
    static OpenCLContext ctx;
    return ctx;
}

Status OpenCLContext::Init() {
    if (initialized) return Status::OK();

    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        LOGE("No OpenCL platforms found");
        return Status::RuntimeError("No OpenCL platforms");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    platform = platforms[0];
    for (auto p : platforms) {
        char name[128];
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        if (std::strstr(name, "NVIDIA") || std::strstr(name, "AMD") ||
            std::strstr(name, "Intel")) {
            platform = p;
            break;
        }
    }

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
                      platform_name, nullptr);

    cl_uint device_count = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
    if (err != CL_SUCCESS || device_count == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
        if (err != CL_SUCCESS || device_count == 0) {
            LOGE("No OpenCL devices found on platform");
            return Status::RuntimeError("No OpenCL devices");
        }
    }

    cl_uint actual_count = 0;
    cl_device_type query_type = CL_DEVICE_TYPE_GPU;
    clGetDeviceIDs(platform, query_type, 0, nullptr, &actual_count);
    if (actual_count == 0) {
        query_type = CL_DEVICE_TYPE_ALL;
        clGetDeviceIDs(platform, query_type, 0, nullptr, &actual_count);
    }

    std::vector<cl_device_id> devices(actual_count);
    clGetDeviceIDs(platform, query_type, actual_count, devices.data(), nullptr);

    device = devices[0];
    for (auto d : devices) {
        cl_device_type t;
        clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(t), &t, nullptr);
        if (t == CL_DEVICE_TYPE_GPU) {
            device = d;
            break;
        }
    }

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                    device_name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_version),
                    device_version, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(total_global_mem), &total_global_mem, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(max_compute_units), &max_compute_units, nullptr);

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
        0
    };

    cl_int ctx_err = 0;
    context = clCreateContext(props, 1, &device, nullptr, nullptr, &ctx_err);
    if (!context || ctx_err != CL_SUCCESS) {
        LOGE("clCreateContext failed: {}", ctx_err);
        return Status::RuntimeError("clCreateContext failed");
    }

    command_queue = clCreateCommandQueueWithProperties(
        context, device, nullptr, &ctx_err);
    if (!command_queue || ctx_err != CL_SUCCESS) {
        clReleaseContext(context);
        context = nullptr;
        LOGE("clCreateCommandQueue failed: {}", ctx_err);
        return Status::RuntimeError("clCreateCommandQueue failed");
    }

    initialized = true;
    LOGI("OpenCL context initialized: platform={} device={} mem={} MB units={}",
         platform_name, device_name, total_global_mem / (1024 * 1024),
         max_compute_units);
    return Status::OK();
}

void OpenCLContext::Destroy() {
    if (!initialized) return;
    if (command_queue) clReleaseCommandQueue(command_queue);
    if (context) clReleaseContext(context);
    command_queue = nullptr;
    context = nullptr;
    device = nullptr;
    platform = nullptr;
    initialized = false;
    LOGI("OpenCL context destroyed");
}

std::vector<DeviceInfo> OpenCLContext::EnumerateDevices() const {
    std::vector<DeviceInfo> devs;
    if (!initialized) return devs;
    DeviceInfo info;
    info.name = device_name;
    info.type = DeviceType::OPENCL;
    info.memory_total = static_cast<size_t>(total_global_mem);
    info.version = device_version;
    info.has_tensor_core = false;
    devs.push_back(info);
    return devs;
}

} // namespace compute_engine
