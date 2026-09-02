#include <compute_engine/compute_engine.h>
#include <compute_engine/memory/resource_tracer.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

using namespace compute_engine;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s\n", msg); \
        tests_failed++; \
    } else { \
        printf("  PASS: %s\n", msg); \
        tests_passed++; \
    } \
} while(0)

static bool NearEqual(float a, float b, float atol = 1e-5f) {
    return std::fabs(a - b) <= atol;
}

// Forward declarations from kernel files (inside compute_engine::cpu namespace)
namespace compute_engine {
namespace cpu {
void ReLUScalar(const float*, float*, int);
void GELUScalar(const float*, float*, int);
void SiLUScalar(const float*, float*, int);
void SoftMaxScalar(const float*, float*, int, int);
void Conv2DScalar(const float*, const float*, float*,
                  int, int, int, int, int, int, int, int, int, int, int);
void AttentionScalar(const float*, const float*, const float*, float*,
                     int, int, int, int);
}
}

// ─── Test CPUID Detection ───
void TestCPUID() {
    printf("\n=== CPUID Detection ===\n");
    auto cpu = BackendRegistry::Instance().GetBestAvailable({DeviceType::CPU});
    CHECK(cpu != nullptr, "CPU backend available");
    if (cpu) {
        CHECK(cpu->device()->IsAvailable(), "CPU device is available");
        CHECK(cpu->device()->name() == "CPU", "CPU device name is 'CPU'");
    }
}

// ─── Test MatMul Scalar ───
void TestMatMulScalar() {
    printf("\n=== MatMul Scalar ===\n");
    int M = 4, N = 4, K = 4;
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) A[i] = (float)(i + 1);
    for (int i = 0; i < K * N; ++i) B[i] = (float)(i + 1);

    reference::MatMul(A.data(), B.data(), C.data(), M, N, K);

    CHECK(NearEqual(C[0], 90.0f), "MatMul C[0][0]=90");
    CHECK(NearEqual(C[1 * N + 1], 228.0f), "MatMul C[1][1]=228");
    CHECK(NearEqual(C[2 * N + 2], 398.0f), "MatMul C[2][2]=398");
    CHECK(NearEqual(C[3 * N + 3], 600.0f), "MatMul C[3][3]=600");
}

// ─── Test LayerNorm Scalar ───
void TestLayerNormScalar() {
    printf("\n=== LayerNorm Scalar ===\n");
    int rows = 2, cols = 4;
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> gamma(cols, 1.0f);
    std::vector<float> beta(cols, 0.0f);
    std::vector<float> output(rows * cols);

    reference::LayerNorm(input.data(), gamma.data(), beta.data(),
                         output.data(), rows, cols, 1e-5f);

    float expected_first = (1.0f - 2.5f) / std::sqrt(1.25f + 1e-5f);
    CHECK(NearEqual(output[0], expected_first, 1e-4f), "LayerNorm row0[0]");
    CHECK(NearEqual(output[4], (5.0f - 6.5f) / std::sqrt(1.25f + 1e-5f), 1e-4f), "LayerNorm row1[0]");
}

// ─── Test RMSNorm Scalar ───
void TestRMSNormScalar() {
    printf("\n=== RMSNorm Scalar ===\n");
    int rows = 1, cols = 4;
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight(cols, 1.0f);
    std::vector<float> output(cols);

    reference::RMSNorm(input.data(), weight.data(), output.data(), rows, cols, 1e-5f);

    float sum_sq = 1.0f + 4.0f + 9.0f + 16.0f;
    float rms = std::sqrt(sum_sq / cols + 1e-5f);
    for (int i = 0; i < cols; ++i) {
        float expected = input[i] / rms;
        CHECK(NearEqual(output[i], expected, 1e-4f), "RMSNorm");
    }
}

// ─── Test ReLU ───
void TestReLU() {
    printf("\n=== ReLU ===\n");
    std::vector<float> input = {-3.0f, -1.0f, 0.0f, 2.0f, 5.0f};
    std::vector<float> output(5);

    cpu::ReLUScalar(input.data(), output.data(), 5);

    CHECK(NearEqual(output[0], 0.0f), "ReLU(-3)=0");
    CHECK(NearEqual(output[1], 0.0f), "ReLU(-1)=0");
    CHECK(NearEqual(output[2], 0.0f), "ReLU(0)=0");
    CHECK(NearEqual(output[3], 2.0f), "ReLU(2)=2");
    CHECK(NearEqual(output[4], 5.0f), "ReLU(5)=5");
}

// ─── Test GELU ───
void TestGELU() {
    printf("\n=== GELU ===\n");
    std::vector<float> input = {0.0f, 1.0f, -1.0f};
    std::vector<float> output(3);

    cpu::GELUScalar(input.data(), output.data(), 3);

    CHECK(NearEqual(output[0], 0.0f, 1e-4f), "GELU(0)=0");
    CHECK(NearEqual(output[1], 0.841192f, 1e-3f), "GELU(1)~0.841");
    CHECK(NearEqual(output[2], -0.158808f, 1e-3f), "GELU(-1)~-0.159");
}

// ─── Test SiLU ───
void TestSiLU() {
    printf("\n=== SiLU ===\n");
    std::vector<float> input = {0.0f, 1.0f, -1.0f};
    std::vector<float> output(3);

    cpu::SiLUScalar(input.data(), output.data(), 3);

    CHECK(NearEqual(output[0], 0.0f, 1e-5f), "SiLU(0)=0");
    CHECK(NearEqual(output[1], 0.731059f, 1e-3f), "SiLU(1)~0.731");
    CHECK(NearEqual(output[2], -0.268941f, 1e-3f), "SiLU(-1)~-0.269");
}

// ─── Test Softmax Scalar ───
void TestSoftmax() {
    printf("\n=== Softmax ===\n");
    int rows = 1, cols = 4;
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(cols);

    cpu::SoftMaxScalar(input.data(), output.data(), rows, cols);

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        sum += output[i];
        CHECK(output[i] > 0.0f, "Softmax output > 0");
    }
    CHECK(NearEqual(sum, 1.0f, 1e-4f), "Softmax sums to 1");
}

// ─── Test Conv2D Scalar vs Reference ───
void TestConv2D() {
    printf("\n=== Conv2D ===\n");
    int N = 1, C = 1, H = 4, W = 4;
    int K = 1, R = 3, S = 3;
    int pad = 0, stride = 1;
    int OH = (H + 2 * pad - R) / stride + 1;
    int OW = (W + 2 * pad - S) / stride + 1;

    std::vector<float> input(N * C * H * W);
    std::vector<float> weight(K * C * R * S);
    std::vector<float> out_ref(N * K * OH * OW);
    std::vector<float> out_cpu(N * K * OH * OW);

    for (int i = 0; i < N * C * H * W; ++i) input[i] = (float)(i + 1);
    for (int i = 0; i < K * C * R * S; ++i) weight[i] = 1.0f;

    reference::Conv2D(input.data(), weight.data(), out_ref.data(),
                      N, C, H, W, K, R, S, pad, pad, stride, stride);
    cpu::Conv2DScalar(input.data(), weight.data(), out_cpu.data(),
                      N, C, H, W, K, R, S, pad, pad, stride, stride);

    bool match = true;
    for (int i = 0; i < N * K * OH * OW; ++i) {
        if (!NearEqual(out_ref[i], out_cpu[i], 1e-5f)) {
            match = false;
            printf("  Mismatch at [%d]: ref=%f cpu=%f\n", i, out_ref[i], out_cpu[i]);
            break;
        }
    }
    CHECK(match, "Conv2D scalar matches reference");
}

// ─── Test Attention Scalar ───
void TestAttention() {
    printf("\n=== Attention ===\n");
    int B = 1, H = 1, N = 4, D = 4;
    std::vector<float> Q(N * D), K(N * D), V(N * D), O(N * D);

    for (int i = 0; i < N * D; ++i) {
        Q[i] = (float)(i + 1) * 0.1f;
        K[i] = (float)(i + 1) * 0.1f;
        V[i] = 1.0f;
    }

    cpu::AttentionScalar(Q.data(), K.data(), V.data(), O.data(), B, H, N, D);

    bool all_ones = true;
    for (int i = 0; i < N * D; ++i) {
        if (!NearEqual(O[i], 1.0f, 1e-4f)) {
            all_ones = false;
            printf("  O[%d]=%f expected 1.0\n", i, O[i]);
            break;
        }
    }
    CHECK(all_ones, "Attention with V=1 yields O=1");
}

// ─── Test Backend API ───
void TestBackendAPI() {
    printf("\n=== Backend API ===\n");
    auto cpu = BackendRegistry::Instance().GetBestAvailable({DeviceType::CPU});
    CHECK(cpu != nullptr, "CPU backend exists");

    Status st = cpu->Initialize();
    CHECK(st.ok(), "CPU backend Initialize");

    st = cpu->Finalize();
    CHECK(st.ok(), "CPU backend Finalize");
}

// ─── Test Tensor Ops ───
void TestTensorOps() {
    printf("\n=== Tensor Operations ===\n");

    Tensor t1({2, 3}, DType::FLOAT32);
    CHECK(t1.shape() == Shape({2, 3}), "Tensor shape [2,3]");
    CHECK(t1.contiguous(), "Tensor contiguous");
    CHECK(t1.num_elements() == 6, "Tensor 6 elements");

    auto t2 = t1.Slice(0, 0, 1);
    CHECK(t2.shape() == Shape({1, 3}), "Slice [0:1] shape [1,3]");

    Tensor t3({2, 3}, DType::FLOAT32);
    auto t4 = t3.Transpose({1, 0});
    CHECK(t4.shape() == Shape({3, 2}), "Transpose shape [3,2]");

    Tensor t5({4, 6}, DType::FLOAT32);
    auto t6 = t5.View(0, {2, 3, 4});
    CHECK(t6.shape() == Shape({2, 3, 4}), "View shape [2,3,4]");
}

// ─── Test Memory ───
void TestMemory() {
    printf("\n=== Memory ===\n");
    CPUAllocator alloc;
    void* ptr = nullptr;
    Status st = alloc.Allocate(&ptr, 1024);
    CHECK(st.ok() && ptr != nullptr, "Allocate 1024 bytes");

    st = alloc.Memset(ptr, 0, 1024);
    CHECK(st.ok(), "Memset 1024 bytes");

    st = alloc.Deallocate(ptr);
    CHECK(st.ok(), "Deallocate");
}

// ─── Test ResourceTracer ───
void TestResourceTracer() {
    printf("\n=== ResourceTracer ===\n");
    auto& tracer = ResourceTracer::Instance();
    CHECK(tracer.NumLive() == 0, "No live resources initially");

    tracer.TrackAlloc((void*)0x1000, ResourceType::TENSOR, __FILE__, __LINE__, "test");
    CHECK(tracer.NumLive() == 1, "1 live resource after track");

    bool freed = tracer.TrackFree((void*)0x1000);
    CHECK(freed, "TrackFree returns true for tracked ptr");
    CHECK(tracer.NumLive() == 0, "No live resources after free");
}

int main() {
    printf("ComputeEngine Unit Tests\n");
    printf("========================\n");

    LogManager::Init("compute_engine_test.log", LogLevel::WARN, false);
    BackendRegistry::InitBuiltinBackends();

    TestCPUID();
    TestTensorOps();
    TestMemory();
    TestResourceTracer();
    TestBackendAPI();
    TestMatMulScalar();
    TestLayerNormScalar();
    TestRMSNormScalar();
    TestReLU();
    TestGELU();
    TestSiLU();
    TestSoftmax();
    TestConv2D();
    TestAttention();

    printf("\n========================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);

    LogManager::Shutdown();
    return tests_failed > 0 ? 1 : 0;
}
