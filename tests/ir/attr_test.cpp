#include <gtest/gtest.h>

#include <cstdint>
#include <variant>
#include <vector>

#include "core/ir/attr.h"

using namespace ie;

TEST(AttrValueTest, HoldsIntAndFloatLists) {
    Attrs attrs;
    attrs["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    attrs["scales"] = std::vector<double>{1.5, 1.5};
    ASSERT_NE(std::get_if<std::vector<int64_t>>(&attrs["pads"]), nullptr);
    ASSERT_NE(std::get_if<std::vector<double>>(&attrs["scales"]), nullptr);
    EXPECT_EQ(std::get<std::vector<int64_t>>(attrs["pads"])[2], 1);
}
