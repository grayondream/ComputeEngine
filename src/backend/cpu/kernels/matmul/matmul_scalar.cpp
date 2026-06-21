#include <cstring>
#include <cstdint>
#include <algorithm>

namespace compute_engine {
namespace cpu {

void MatMulScalar(const float* A, const float* B, float* C,
                  int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void MatMulScalarTiled(const float* A, const float* B, float* C,
                       int M, int N, int K, int tile_size) {
    // Zero C
    std::memset(C, 0, M * N * sizeof(float));

    for (int mi = 0; mi < M; mi += tile_size) {
        int me = std::min(mi + tile_size, M);
        for (int ni = 0; ni < N; ni += tile_size) {
            int ne = std::min(ni + tile_size, N);
            for (int ki = 0; ki < K; ki += tile_size) {
                int ke = std::min(ki + tile_size, K);
                for (int m = mi; m < me; ++m) {
                    for (int k = ki; k < ke; ++k) {
                        float a_val = A[m * K + k];
                        for (int n = ni; n < ne; ++n) {
                            C[m * N + n] += a_val * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
}

} // namespace cpu
} // namespace compute_engine
