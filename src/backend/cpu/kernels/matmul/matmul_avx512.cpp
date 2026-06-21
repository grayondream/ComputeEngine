#ifdef CE_HAS_AVX512
#include <immintrin.h>

namespace compute_engine {
namespace cpu {

void MatMulAVX512(const float* A, const float* B, float* C,
                  int M, int N, int K) {
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 16;

    __builtin_memset(C, 0, M * N * sizeof(float));

    for (int mi = 0; mi < M; mi += TILE_M) {
        int me = mi + TILE_M <= M ? mi + TILE_M : M;
        for (int ni = 0; ni < N; ni += TILE_N) {
            int ne = ni + TILE_N <= N ? ni + TILE_N : N;
            __m512 c_vec[TILE_M];
            for (int m = 0; m < me - mi; ++m)
                c_vec[m] = _mm512_loadu_ps(&C[(mi + m) * N + ni]);

            for (int k = 0; k < K; ++k) {
                for (int m = 0; m < me - mi; ++m) {
                    __m512 a_vec = _mm512_set1_ps(A[(mi + m) * K + k]);
                    __m512 b_vec = _mm512_loadu_ps(&B[k * N + ni]);
                    c_vec[m] = _mm512_fmadd_ps(a_vec, b_vec, c_vec[m]);
                }
            }

            for (int m = 0; m < me - mi; ++m)
                _mm512_storeu_ps(&C[(mi + m) * N + ni], c_vec[m]);
        }
    }
}

} // namespace cpu
} // namespace compute_engine
#endif
