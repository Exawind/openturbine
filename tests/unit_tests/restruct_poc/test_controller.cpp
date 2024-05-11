#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/utilities/discon.h"

namespace openturbine::restruct_poc::tests {

TEST(ControllerTest, ClampFunction) {
    // test case 1: v is less than v_min
    float v = 1.0;
    float v_min = 2.0;
    float v_max = 3.0;
    float expected = 2.0;
    float actual = openturbine::util::clamp(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);

    // test case 2: v is greater than v_max
    v = 4.0;
    expected = 3.0;
    actual = openturbine::util::clamp(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);

    // test case 3: v is between v_min and v_max
    v = 2.5;
    expected = 2.5;
    actual = openturbine::util::clamp(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);
}

}  // namespace openturbine::restruct_poc::tests
