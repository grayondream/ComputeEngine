#include <cmath>
#include <cstring>

namespace compute_engine {
namespace cpu {

void LayerNormScalar(const float* input, const float* gamma, const float* beta,
                     float* output, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        float* out = output + r * cols;

        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += row[c];
        mean /= cols;

        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = row[c] - mean;
            var += d * d;
        }
        var /= cols;

        float inv = 1.0f / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c) {
            float norm = (row[c] - mean) * inv;
            out[c] = gamma[c] * norm + beta[c];
        }
    }
}

void RMSNormScalar(const float* input, const float* weight, float* output,
                   int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        float* out = output + r * cols;

        float sum_sq = 0.0f;
        for (int c = 0; c < cols; ++c)
            sum_sq += row[c] * row[c];

        float rms = std::sqrt(sum_sq / cols + eps);
        float inv_rms = 1.0f / rms;
        for (int c = 0; c < cols; ++c)
            out[c] = row[c] * inv_rms * weight[c];
    }
}

void BatchNormScalar(const float* input, const float* gamma, const float* beta,
                     const float* mean, const float* variance,
                     float* output, int N, int C, int H, int W, float eps) {
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            float inv = 1.0f / std::sqrt(variance[c] + eps);
            float scale = gamma[c] * inv;
            float shift = -mean[c] * scale + beta[c];
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    output[idx] = input[idx] * scale + shift;
                }
        }
}

void SoftMaxScalar(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        float* out = output + r * cols;

        float max_val = row[0];
        for (int c = 1; c < cols; ++c)
            max_val = std::max(max_val, row[c]);

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            out[c] = std::exp(row[c] - max_val);
            sum += out[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c)
            out[c] *= inv_sum;
    }
}

} // namespace cpu
} // namespace compute_engine
