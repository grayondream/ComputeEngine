#ifdef CE_HAS_AVX2
#include <immintrin.h>
#include <cstring>
#include <algorithm>

namespace compute_engine {
namespace cpu {

static void Im2Col(const float* data_im, float* data_col,
                   int channels, int height, int width,
                   int kernel_h, int kernel_w,
                   int pad_h, int pad_w,
                   int stride_h, int stride_w,
                   int output_h, int output_w) {
    for (int c = 0; c < channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int out_row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        int ih = oh * stride_h - pad_h + kh;
                        int iw = ow * stride_w - pad_w + kw;
                        float val = (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            ? data_im[(c * height + ih) * width + iw]
                            : 0.0f;
                        data_col[out_row * output_h * output_w + oh * output_w + ow] = val;
                    }
                }
            }
        }
    }
}

static void MatMulAVX_GEMM(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    constexpr int TILE_M = 8;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 8;

    std::memset(C, 0, M * N * sizeof(float));

    for (int mi = 0; mi < M; mi += TILE_M) {
        int me = std::min(mi + TILE_M, M);
        for (int ki = 0; ki < K; ki += TILE_K) {
            int ke = std::min(ki + TILE_K, K);
            for (int ni = 0; ni < N; ni += TILE_N) {
                int ne = std::min(ni + TILE_N, N);

                __m256 c_acc[TILE_M / 1][TILE_N / 8];
                for (int m = 0; m < me - mi; ++m)
                    for (int n = 0; n < (ne - ni + 7) / 8; ++n)
                        c_acc[m][n] = _mm256_loadu_ps(&C[(mi + m) * N + ni + n * 8]);

                for (int k = ki; k < ke; ++k) {
                    __m256 b_vec[TILE_N / 8];
                    for (int n = 0; n < (ne - ni + 7) / 8; ++n)
                        b_vec[n] = _mm256_loadu_ps(&B[k * N + ni + n * 8]);

                    for (int m = 0; m < me - mi; ++m) {
                        __m256 a_vec = _mm256_broadcast_ss(&A[(mi + m) * K + k]);
                        for (int n = 0; n < (ne - ni + 7) / 8; ++n)
                            c_acc[m][n] = _mm256_fmadd_ps(a_vec, b_vec[n], c_acc[m][n]);
                    }
                }

                for (int m = 0; m < me - mi; ++m)
                    for (int n = 0; n < (ne - ni + 7) / 8; ++n)
                        _mm256_storeu_ps(&C[(mi + m) * N + ni + n * 8], c_acc[m][n]);
            }
        }
    }
}

void Conv2DAVX(const float* input, const float* weight, float* output,
               int N, int C, int H, int W,
               int K, int R, int S,
               int pad_h, int pad_w,
               int stride_h, int stride_w) {
    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (W + 2 * pad_w - S) / stride_w + 1;

    int col_rows = C * R * S;
    int col_cols = OH * OW;

    float* col_buf = static_cast<float*>(std::malloc(col_rows * col_cols * sizeof(float)));
    float* output_reshaped = static_cast<float*>(std::malloc(K * col_cols * sizeof(float)));

    std::memset(output, 0, N * K * OH * OW * sizeof(float));

    for (int n = 0; n < N; ++n) {
        const float* input_n = input + n * C * H * W;

        Im2Col(input_n, col_buf, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w, OH, OW);

        MatMulAVX_GEMM(weight, col_buf, output_reshaped, K, col_cols, C * R * S);

        for (int k = 0; k < K; ++k)
            std::memcpy(output + (n * K + k) * OH * OW,
                        output_reshaped + k * col_cols,
                        OH * OW * sizeof(float));
    }

    std::free(col_buf);
    std::free(output_reshaped);
}

} // namespace cpu
} // namespace compute_engine
#endif // CE_HAS_AVX2
