#include <cmath>
#include <algorithm>

namespace compute_engine {
namespace cpu {

void ReLUScalar(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i)
        output[i] = std::max(0.0f, input[i]);
}

void LeakyReLUScalar(const float* input, float* output, int size, float alpha) {
    for (int i = 0; i < size; ++i)
        output[i] = input[i] >= 0.0f ? input[i] : alpha * input[i];
}

void GELUScalar(const float* input, float* output, int size) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float t = c * (x + coeff * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(t));
    }
}

void FastGELUScalar(const float* input, float* output, int size) {
    const float coeff = 1.702f;
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-coeff * x));
        output[i] = x * sigmoid;
    }
}

void SiLUScalar(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

void SigmoidScalar(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i)
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
}

void TanhScalar(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i)
        output[i] = std::tanh(input[i]);
}

void SwiGLUScalar(const float* input, const float* gate, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float g = gate[i];
        output[i] = x * (g / (1.0f + std::exp(-g)));
    }
}

} // namespace cpu
} // namespace compute_engine
