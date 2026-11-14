#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/memory/malloc_allocator.h"
#include "data/tiny_cnn_reference.h"
#include "engine/traditional/traditional_engine.h"

using namespace ie;

namespace {

Tensor make_tensor(const Shape& shape, std::span<const float> values) {
    auto alloc = std::make_shared<MallocAllocator>();
    auto tensor = Tensor::allocate(shape, DType::F32, alloc);
    EXPECT_TRUE(tensor.has_value());
    Tensor result = std::move(*tensor);
    std::copy(values.begin(), values.end(), result.data<float>());
    return result;
}

std::string tiny_cnn_path() {
    return std::string(IE_TEST_DATA_DIR) + "/tiny_cnn.onnx";
}

}  // namespace

TEST(TraditionalEngineTest, LoadsAndRunsTinyCnn) {
    TraditionalEngine engine;
    ASSERT_TRUE(engine.load_model(tiny_cnn_path(), ModelFormat::ONNX)
                    .has_value());

    auto input = make_tensor(Shape{{1, 3, 8, 8}}, kTinyCnnInput);
    auto out = engine.infer({input});
    ASSERT_TRUE(out.has_value()) << out.error().message;
    ASSERT_EQ(out->shape.dims, (std::vector<int64_t>{1, 3}));
    for (std::size_t i = 0; i < std::size(kTinyCnnExpected); ++i) {
        EXPECT_NEAR(out->data<float>()[i], kTinyCnnExpected[i], 1e-4f);
    }
}

TEST(TraditionalEngineTest, RepeatedInferKeepsWeightsAlive) {
    TraditionalEngine engine;
    ASSERT_TRUE(engine.load_model(tiny_cnn_path(), ModelFormat::ONNX)
                    .has_value());
    auto input = make_tensor(Shape{{1, 3, 8, 8}}, kTinyCnnInput);

    auto first = engine.infer({input});
    ASSERT_TRUE(first.has_value()) << first.error().message;
    auto second = engine.infer({input});
    ASSERT_TRUE(second.has_value()) << second.error().message;
    for (std::size_t i = 0; i < std::size(kTinyCnnExpected); ++i) {
        EXPECT_NEAR(second->data<float>()[i], first->data<float>()[i], 0.0f);
        EXPECT_NEAR(second->data<float>()[i], kTinyCnnExpected[i], 1e-4f);
    }
}

TEST(TraditionalEngineTest, BatchMatchesSingleAndRejectsWrongFormat) {
    TraditionalEngine engine;
    ASSERT_TRUE(engine.load_model(tiny_cnn_path(), ModelFormat::ONNX)
                    .has_value());

    auto input = make_tensor(Shape{{1, 3, 8, 8}}, kTinyCnnInput);
    std::vector<float> scaled(kTinyCnnInput,
                              kTinyCnnInput + std::size(kTinyCnnInput));
    for (float& value : scaled) {
        value *= 0.5f;
    }
    auto other = make_tensor(Shape{{1, 3, 8, 8}}, scaled);

    auto batch = engine.infer_batch({{input}, {other}});
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->size(), 2u);

    auto single = engine.infer({input});
    ASSERT_TRUE(single.has_value()) << single.error().message;
    auto other_single = engine.infer({other});
    ASSERT_TRUE(other_single.has_value()) << other_single.error().message;

    for (std::size_t i = 0; i < std::size(kTinyCnnExpected); ++i) {
        EXPECT_NEAR(batch->at(0).data<float>()[i],
                    single->data<float>()[i], 0.0f);
        EXPECT_NEAR(batch->at(1).data<float>()[i],
                    other_single->data<float>()[i], 0.0f);
    }

    auto empty = engine.infer_batch({});
    ASSERT_TRUE(empty.has_value());
    EXPECT_TRUE(empty->empty());

    TraditionalEngine wrong_format;
    EXPECT_EQ(wrong_format.load_model(tiny_cnn_path(), ModelFormat::TFLITE)
                  .error().code,
              Error::Code::Unsupported);
}

TEST(TraditionalEngineTest, RejectsWrongInputCount) {
    TraditionalEngine engine;
    ASSERT_TRUE(engine.load_model(tiny_cnn_path(), ModelFormat::ONNX)
                    .has_value());
    auto input = make_tensor(Shape{{1, 3, 8, 8}}, kTinyCnnInput);

    auto none = engine.infer({});
    ASSERT_FALSE(none.has_value());
    EXPECT_EQ(none.error().code, Error::Code::InvalidArgument);

    auto too_many = engine.infer({input, input});
    ASSERT_FALSE(too_many.has_value());
    EXPECT_EQ(too_many.error().code, Error::Code::InvalidArgument);

    auto bad_sample = engine.infer_batch({{input}, {}});
    ASSERT_FALSE(bad_sample.has_value());
    EXPECT_EQ(bad_sample.error().code, Error::Code::InvalidArgument);
}
