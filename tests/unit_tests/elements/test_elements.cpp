#include <gtest/gtest.h>

#include "src/elements/elements.hpp"

namespace openturbine::tests {

TEST(ElementsTest, ExpectThrowOnNullElements) {
    EXPECT_THROW(Elements(), std::invalid_argument);
    EXPECT_THROW(Elements(nullptr, nullptr), std::invalid_argument);
}

TEST(ElementsTest, ConstructorWithBeams) {
    auto beams = std::make_shared<Beams>(1, 2, 2);  // 1 beam element with 2 nodes, 2 qps
    Elements elements(beams);
    EXPECT_EQ(elements.beams->num_elems, 1);
    EXPECT_EQ(elements.masses, nullptr);
}

TEST(ElementsTest, ConstructorWithMasses) {
    auto masses = std::make_shared<Masses>(1);  // 1 mass element
    Elements elements(nullptr, masses);
    EXPECT_EQ(elements.beams, nullptr);
    EXPECT_EQ(elements.masses->num_elems, 1);
}

TEST(ElementsTest, ConstructorWithBothElements) {
    auto beams = std::make_shared<Beams>(1, 2, 2);  // 1 beam element with 2 nodes, 2 qps
    auto masses = std::make_shared<Masses>(1);      // 1 mass element
    Elements elements(beams, masses);
    EXPECT_EQ(elements.beams->num_elems, 1);
    EXPECT_EQ(elements.masses->num_elems, 1);
}

}  // namespace openturbine::tests