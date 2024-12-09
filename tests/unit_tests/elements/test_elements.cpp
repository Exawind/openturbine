#include <gtest/gtest.h>

#include "src/elements/elements.hpp"
#include "src/step/step.hpp"

namespace openturbine::tests {

TEST(ElementsTest, ExpectThrowOnNullElements) {
    EXPECT_THROW(Elements(), std::invalid_argument);
    EXPECT_THROW(Elements(nullptr, nullptr), std::invalid_argument);
}

TEST(ElementsTest, ConstructorWithBeams) {
    auto beams = std::make_shared<Beams>(1, 2, 2);  // 1 beam element with 2 nodes, 2 qps
    Elements elements(beams);                       // No mass elements in the model
    EXPECT_EQ(elements.NumElementsInSystem(), 1);
    EXPECT_EQ(elements.beams->num_elems, 1);
    EXPECT_EQ(elements.masses, nullptr);
}

TEST(ElementsTest, ConstructorWithMasses) {
    auto masses = std::make_shared<Masses>(1);  // 1 mass element
    Elements elements(nullptr, masses);         // No beam elements in the model
    EXPECT_EQ(elements.NumElementsInSystem(), 1);
    EXPECT_EQ(elements.beams, nullptr);
    EXPECT_EQ(elements.masses->num_elems, 1);
}

TEST(ElementsTest, ConstructorWithBothElements) {
    auto beams = std::make_shared<Beams>(1, 2, 2);  // 1 beam element with 2 nodes, 2 qps
    auto masses = std::make_shared<Masses>(1);      // 1 mass element
    Elements elements(beams, masses);
    EXPECT_EQ(elements.NumElementsInSystem(), 2);
    EXPECT_EQ(elements.beams->num_elems, 1);
    EXPECT_EQ(elements.masses->num_elems, 1);
}

TEST(ElementsTest, NumberOfNodesPerElementBeamsOnly) {
    auto beams = std::make_shared<Beams>(2, 3, 2);  // 2 beam elements with 3 nodes each
    Kokkos::deep_copy(beams->num_nodes_per_element, 3U);
    Elements elements{beams};

    EXPECT_EQ(elements.NumElementsInSystem(), 2);

    auto host_nodes_per_elem = Kokkos::create_mirror_view(elements.NumberOfNodesPerElement());
    Kokkos::deep_copy(host_nodes_per_elem, elements.NumberOfNodesPerElement());

    EXPECT_EQ(host_nodes_per_elem(0), 3);  // First beam element has 3 nodes
    EXPECT_EQ(host_nodes_per_elem(1), 3);  // Second beam element has 3 nodes
}

TEST(ElementsTest, NumberOfNodesPerElementMassesOnly) {
    auto masses = std::make_shared<Masses>(3);  // 3 mass elements
    Elements elements{nullptr, masses};

    EXPECT_EQ(elements.NumElementsInSystem(), 3);

    auto host_nodes_per_elem = Kokkos::create_mirror_view(elements.NumberOfNodesPerElement());
    Kokkos::deep_copy(host_nodes_per_elem, elements.NumberOfNodesPerElement());

    // All mass elements should have 1 node
    EXPECT_EQ(host_nodes_per_elem(0), 1);
    EXPECT_EQ(host_nodes_per_elem(1), 1);
    EXPECT_EQ(host_nodes_per_elem(2), 1);
}

TEST(ElementsTest, NumberOfNodesPerElementMixedElements) {
    auto beams = std::make_shared<Beams>(2, 4, 2);  // 2 beam elements with 4 nodes each
    Kokkos::deep_copy(beams->num_nodes_per_element, 4U);
    auto masses = std::make_shared<Masses>(2);  // 2 mass elements
    Elements elements{beams, masses};

    EXPECT_EQ(elements.NumElementsInSystem(), 4);

    auto host_nodes_per_elem = Kokkos::create_mirror_view(elements.NumberOfNodesPerElement());
    Kokkos::deep_copy(host_nodes_per_elem, elements.NumberOfNodesPerElement());

    // First two elements are beams
    EXPECT_EQ(host_nodes_per_elem(0), 4);
    EXPECT_EQ(host_nodes_per_elem(1), 4);

    // Last two elements are masses
    EXPECT_EQ(host_nodes_per_elem(2), 1);
    EXPECT_EQ(host_nodes_per_elem(3), 1);
}

}  // namespace openturbine::tests
