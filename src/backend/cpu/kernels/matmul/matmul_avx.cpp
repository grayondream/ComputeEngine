#ifdef CE_HAS_AVX2
#include <immintrin.h>
#include <cstring>
#include <algorithm>

namespace compute_engine {
namespace cpu {

static inline void PackB(const float* B, float* B_packed, int N, int K, int tile_n, int tile_k) {
    for (int ni = 0; ni < tile_n; ni += 8) {
        for (int ki = 0; ki < tile_k; ++ki) {
            for (int j = 0; j < 8 && ni + j < N; ++j) {
                B_packed[ni * tile_k + ki * 8 + j] = B[(ki) * N + ni + j];
            }
            for (int j = tile_k * 8; j < 8 * tile_k; ++j) {
                B_packed[ni * tile_k + ki * 8 + j - tile_k * 8 + ki * 8] = 0;
            }
        }
    }
}

void MatMulAVX(const float* A, const float* B, float* C,
               int M, int N, int K) {
    constexpr int TILE_M = 8;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 8;

    std::memset(C, 0, M * N * sizeof(float));

    float B_packed[TILE_N * TILE_K];

    for (int mi = 0; mi < M; mi += TILE_M) {
        int me = std::min(mi + TILE_M, M);
        for (int ki = 0; ki < K; ki += TILE_K) {
            int ke = std::min(ki + TILE_K, K);
            for (int ni = 0; ni < N; ni += TILE_N) {
                int ne = std::min(ni + TILE_N, N);

                __m256 c[TILE_M / 1][TILE_N / 8];
                for (int m = 0; m < me - mi; ++m) {
                    for (int n = 0; n < (ne - ni + 7) / 8; ++n) {
                        c[m][n] = _mm256_loadu_ps(&C[(mi + m) * N + ni + n * 8]);
                    }
                }

                for (int k = ki; k < ke; ++k) {
                    __m256 b_vec[TILE_N / 8];
                    for (int n = 0; n < (ne - ni + 7) / 8; ++n) {
                        b_vec[n] = _mm256_loadu_ps(&B[k * N + ni + n * 8]);
                    }

                    for (int m = 0; m < me - mi; ++m) {
                        __m256 a_vec = _mm256_broadcast_ss(&A[(mi + m) * K + k]);
                        for (int n = 0; n < (ne - ni + 7) / 8; ++n) {
                            c[m][n] = _mm256_fmadd_ps(a_vec, b_vec[n], c[m][n]);
                        }
                    }
                }

                for (int m = 0; m < me - mi; ++m) {
                    for (int n = 0; n < (ne - ni + 7) / 8; ++n) {
                        _mm256_storeu_ps(&C[(mi + m) * N + ni + n * 8], c[m][n]);
                    }
                }
            }
        }
    }
}

} // namespace cpu
} // namespace compute_engine

#endif // CE_HAS_AVX2
