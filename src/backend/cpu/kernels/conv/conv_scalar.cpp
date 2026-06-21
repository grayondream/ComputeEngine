// Conv2D Scalar - reference implementation
#include <cstring>

namespace compute_engine {
namespace cpu {

void Conv2DScalar(const float* input, const float* weight, float* output,
                  int N, int C, int H, int W,
                  int K, int R, int S,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w) {
    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (W + 2 * pad_w - S) / stride_w + 1;

    __builtin_memset(output, 0, N * K * OH * OW * sizeof(float));

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int ih = oh * stride_h - pad_h + r;
                                int iw = ow * stride_w - pad_w + s;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    sum += input[((n * C + c) * H + ih) * W + iw]
                                         * weight[((k * C + c) * R + r) * S + s];
                                }
                            }
                        }
                    }
                    output[((n * K + k) * OH + oh) * OW + ow] = sum;
                }
            }
        }
    }
}

} // namespace cpu
} // namespace compute_engine
