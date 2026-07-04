#include "compute_engine/backend/opencl_backend.h"
#include "compute_engine/platform/logger.h"
#include <cstring>
#include <vector>

namespace compute_engine {

// ─── OpenCLAllocator ───

Status OpenCLAllocator::Allocate(void** ptr, size_t size) {
    if (size == 0) { *ptr = nullptr; return Status::OK(); }
    OpenCLContext& ctx = OpenCLContext::Instance();
    if (!ctx.initialized) return Status::RuntimeError("OpenCL not initialized");

    cl_int err = 0;
    cl_mem mem = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, size,
                                nullptr, &err);
    if (!mem || err != CL_SUCCESS) {
        return Status::RuntimeError("clCreateBuffer failed");
    }
    *ptr = mem;
    LOGI("OpenCLAllocator::Allocate size={}", size);
    return Status::OK();
}

Status OpenCLAllocator::Deallocate(void* ptr) {
    if (!ptr) return Status::OK();
    cl_mem mem = static_cast<cl_mem>(ptr);
    clReleaseMemObject(mem);
    LOGI("OpenCLAllocator::Deallocate");
    return Status::OK();
}

Status OpenCLAllocator::Copy(void* dst, const void* src,
                              size_t size, CopyDirection dir) {
    if (!dst || !src || size == 0) return Status::OK();
    OpenCLContext& ctx = OpenCLContext::Instance();

    cl_bool blocking = CL_TRUE;
    cl_int err = CL_SUCCESS;

    switch (dir) {
        case CopyDirection::H2D:
            err = clEnqueueWriteBuffer(ctx.command_queue,
                                       static_cast<cl_mem>(dst), blocking,
                                       0, size, src, 0, nullptr, nullptr);
            break;
        case CopyDirection::D2H:
            err = clEnqueueReadBuffer(ctx.command_queue,
                                      static_cast<cl_mem>(const_cast<void*>(src)), blocking,
                                      0, size, dst, 0, nullptr, nullptr);
            break;
        case CopyDirection::D2D:
            err = clEnqueueCopyBuffer(ctx.command_queue,
                                      static_cast<cl_mem>(const_cast<void*>(src)),
                                      static_cast<cl_mem>(dst),
                                      0, 0, size, 0, nullptr, nullptr);
            break;
    }

    if (err != CL_SUCCESS) {
        return Status::RuntimeError("OpenCL copy failed");
    }
    clFinish(ctx.command_queue);
    return Status::OK();
}

Status OpenCLAllocator::Memset(void* ptr, int value, size_t size) {
    if (!ptr || size == 0) return Status::OK();
    OpenCLContext& ctx = OpenCLContext::Instance();

    std::vector<char> pattern(size, static_cast<char>(value));
    cl_int err = clEnqueueWriteBuffer(ctx.command_queue,
                                      static_cast<cl_mem>(ptr), CL_TRUE,
                                      0, size, pattern.data(),
                                      0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return Status::RuntimeError("OpenCL memset failed");
    }
    clFinish(ctx.command_queue);
    return Status::OK();
}

Status OpenCLAllocator::CreateTensor(TensorPtr& tensor,
                                      const Shape& shape, DType dtype) {
    tensor = std::make_shared<Tensor>(shape, dtype);
    return Status::OK();
}

// ─── OpenCLKernelExecutor ───

Status OpenCLKernelExecutor::CompileKernel(
    const std::string& name, const std::vector<uint8_t>& source,
    KernelHandle* handle) {
    OpenCLContext& ctx = OpenCLContext::Instance();
    if (!ctx.initialized) return Status::RuntimeError("OpenCL not initialized");

    const char* src_str = reinterpret_cast<const char*>(source.data());
    size_t src_len = source.size();

    cl_int err = 0;
    cl_program program = clCreateProgramWithSource(
        ctx.context, 1, &src_str, &src_len, &err);
    if (!program || err != CL_SUCCESS) {
        LOGE("clCreateProgramWithSource failed for kernel '{}'", name);
        return Status::RuntimeError("clCreateProgramWithSource failed");
    }

    err = clBuildProgram(program, 1, &ctx.device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char build_log[4096] = {};
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG,
                              sizeof(build_log), build_log, nullptr);
        LOGE("OpenCL kernel '{}' build failed: {}", name, build_log);
        clReleaseProgram(program);
        return Status::RuntimeError("clBuildProgram failed: " +
                                     std::string(build_log));
    }

    cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
    if (!kernel || err != CL_SUCCESS) {
        LOGE("clCreateKernel '{}' failed", name);
        clReleaseProgram(program);
        return Status::RuntimeError("clCreateKernel failed");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    KernelHandle id = kernels_.size() + 1;
    kernels_.push_back({program, kernel, name});
    if (handle) *handle = id;
    LOGI("OpenCL kernel compiled: '{}' handle={}", name, id);
    return Status::OK();
}

Status OpenCLKernelExecutor::LaunchKernel(
    KernelHandle handle, const std::vector<BufferArg>& args,
    const LaunchConfig& config, int stream_id) {
    CE_UNUSED(stream_id);
    OpenCLContext& ctx = OpenCLContext::Instance();
    if (!ctx.initialized) return Status::RuntimeError("OpenCL not initialized");

    OpenCLKernelEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t idx = handle - 1;
        if (idx >= kernels_.size()) {
            return Status::RuntimeError("Invalid kernel handle");
        }
        entry = &kernels_[idx];
    }

    cl_int err = CL_SUCCESS;
    for (size_t i = 0; i < args.size(); i++) {
        cl_mem mem = static_cast<cl_mem>(args[i].tensor->data());
        err = clSetKernelArg(entry->kernel, static_cast<cl_uint>(i),
                             sizeof(cl_mem), &mem);
        if (err != CL_SUCCESS) {
            LOGE("clSetKernelArg failed for arg {}", i);
            return Status::RuntimeError("clSetKernelArg failed");
        }
    }

    size_t global[3] = {
        static_cast<size_t>(config.grid[0] * config.block[0]),
        static_cast<size_t>(config.grid[1] * config.block[1]),
        static_cast<size_t>(config.grid[2] * config.block[2])
    };
    size_t local[3] = {
        static_cast<size_t>(config.block[0]),
        static_cast<size_t>(config.block[1]),
        static_cast<size_t>(config.block[2])
    };

    err = clEnqueueNDRangeKernel(ctx.command_queue, entry->kernel, 3,
                                 nullptr, global, local,
                                 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        LOGE("clEnqueueNDRangeKernel failed: {}", err);
        return Status::RuntimeError("clEnqueueNDRangeKernel failed");
    }

    clFinish(ctx.command_queue);
    LOGI("OpenCL kernel launched: '{}' global={}x{}x{} local={}x{}x{}",
         entry->name, global[0], global[1], global[2],
         local[0], local[1], local[2]);
    return Status::OK();
}

Status OpenCLKernelExecutor::ReleaseKernel(KernelHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t idx = handle - 1;
    if (idx >= kernels_.size()) return Status::OK();
    auto& k = kernels_[idx];
    clReleaseKernel(k.kernel);
    clReleaseProgram(k.program);
    LOGI("OpenCL kernel released: '{}'", k.name);
    return Status::OK();
}

Status OpenCLKernelExecutor::BeginGraphCapture(int) {
    return Status::NotImplemented("OpenCL does not support graph capture");
}
Status OpenCLKernelExecutor::EndGraphCapture(GraphHandle*) {
    return Status::NotImplemented("OpenCL does not support graph capture");
}
Status OpenCLKernelExecutor::LaunchGraph(GraphHandle) {
    return Status::NotImplemented("OpenCL does not support graph capture");
}

} // namespace compute_engine
