#include <gtest/gtest.h>

#include "core/base/error.h"
#include "core/base/shape.h"
#include "core/base/types.h"

using namespace ie;

TEST(ShapeTest, NumelAndRowMajorStride) {
    Shape s{{2, 3, 4}};
    EXPECT_EQ(s.numel(), 24);
    EXPECT_EQ(s.stride(0), 12);
    EXPECT_EQ(s.stride(1), 4);
    EXPECT_EQ(s.stride(2), 1);
}

TEST(ShapeTest, Equality) {
    EXPECT_TRUE((Shape{{2, 3}} == Shape{{2, 3}}));
    EXPECT_FALSE((Shape{{2, 3}} == Shape{{3, 2}}));
}

TEST(BaseTypesTest, DTypeSizes) {
    EXPECT_EQ(dtype_size(DType::F32), 4u);
    EXPECT_EQ(dtype_size(DType::F16), 2u);
    EXPECT_EQ(dtype_size(DType::I8), 1u);
}

TEST(ErrorTest, FactoriesCarryCodeAndMessage) {
    auto e = Error::invalid_argument("bad shape");
    EXPECT_EQ(e.code, Error::Code::InvalidArgument);
    EXPECT_EQ(e.message, "bad shape");
    Result<int> r = e;
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, Error::Code::InvalidArgument);
}
