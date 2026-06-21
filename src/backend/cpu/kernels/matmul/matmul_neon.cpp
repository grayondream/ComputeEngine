#ifdef CE_HAS_NEON
#include <arm_neon.h>

namespace compute_engine {
namespace cpu {

void MatMulNEON(const float* A, const float* B, float* C,
                int M, int N, int K) {
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 8;

    __builtin_memset(C, 0, M * N * sizeof(float));

    for (int mi = 0; mi < M; mi += TILE_M) {
        int me = mi + TILE_M <= M ? mi + TILE_M : M;
        for (int ni = 0; ni < N; ni += TILE_N) {
            int ne = ni + TILE_N <= N ? ni + TILE_N : N;
            float32x4_t c_vec[TILE_M];
            for (int m = 0; m < me - mi; ++m)
                c_vec[m] = vld1q_f32(&C[(mi + m) * N + ni]);

            for (int k = 0; k < K; ++k) {
                float32x4_t b_lo = vld1q_f32(&B[k * N + ni]);
                float32x4_t b_hi = (ne - ni > 4) ? vld1q_f32(&B[k * N + ni + 4]) : vdupq_n_f32(0);
                for (int m = 0; m < me - mi; ++m) {
                    float32x4_t a_vec = vdupq_n_f32(A[(mi + m) * K + k]);
                    c_vec[m] = vmlaq_f32(c_vec[m], a_vec, b_lo);
                    if (ne - ni > 4) {
                        float32x4_t c_hi = (m == 0) ? vld1q_f32(&C[(mi + m) * N + ni + 4]) :
                                                      vdupq_n_f32(0);
                    }
                }
            }

            for (int m = 0; m < me - mi; ++m)
                vst1q_f32(&C[(mi + m) * N + ni], c_vec[m]);
        }
    }
}

} // namespace cpu
} // namespace compute_engine
#endif
