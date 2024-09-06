#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/populate_sparse_indices.hpp"

namespace openturbine::tests {

TEST(PopulateSparseIndices, SingleElement) {
    constexpr auto num_nodes = size_t{5U};
    constexpr auto num_dof = num_nodes * 6U;

    auto num_nodes_per_element = Kokkos::View<size_t[1]>("num_nodes_per_element");
    auto num_nodes_per_element_host = Kokkos::create_mirror(num_nodes_per_element);
    num_nodes_per_element_host(0) = num_nodes;
    Kokkos::deep_copy(num_nodes_per_element, num_nodes_per_element_host);

    auto node_state_indices = Kokkos::View<size_t[1][num_nodes]>("node_state_indices");
    auto node_state_indices_host_data = std::array<size_t, num_nodes>{0U, 1U, 2U, 3U, 4U};
    auto node_state_indices_host =
        Kokkos::View<size_t[1][num_nodes], Kokkos::HostSpace>(node_state_indices_host_data.data());
    auto node_state_indices_mirror = Kokkos::create_mirror(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);
    auto indices = Kokkos::View<int[num_dof * num_dof]>("indices");

    Kokkos::parallel_for(
        1, PopulateSparseIndices{num_nodes_per_element, node_state_indices, indices}
    );

    auto indices_host = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_host, indices);
    for (auto row = 0U; row < num_dof; ++row) {
        for (auto column = 0U; column < num_dof; ++column) {
            ASSERT_EQ(indices_host(row * num_dof + column), column);
        }
    }
}

TEST(PopulateSparseIndices, TwoElements) {
    constexpr auto elem1_num_nodes = size_t{5U};
    constexpr auto elem1_num_dof = elem1_num_nodes * 6U;
    constexpr auto elem2_num_nodes = size_t{3U};
    constexpr auto elem2_num_dof = elem2_num_nodes * 6U;
    constexpr auto max_num_nodes = elem1_num_nodes;

    auto num_nodes_per_element = Kokkos::View<size_t[2]>("num_nodes_per_element");
    auto num_nodes_per_element_host = Kokkos::create_mirror(num_nodes_per_element);
    num_nodes_per_element_host(0) = elem1_num_nodes;
    num_nodes_per_element_host(1) = elem2_num_nodes;
    Kokkos::deep_copy(num_nodes_per_element, num_nodes_per_element_host);

    auto node_state_indices = Kokkos::View<size_t[2][max_num_nodes]>("node_state_indices");
    auto node_state_indices_host_data =
        std::array<size_t, size_t{2U} * max_num_nodes>{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 0U, 0U};
    auto node_state_indices_host =
        Kokkos::View<size_t[2][max_num_nodes], Kokkos::HostSpace>(node_state_indices_host_data.data()
        );
    auto node_state_indices_mirror = Kokkos::create_mirror(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);
    auto indices =
        Kokkos::View<int[elem1_num_dof * elem1_num_dof + elem2_num_dof * elem2_num_dof]>("indices");

    Kokkos::parallel_for(
        1, PopulateSparseIndices{num_nodes_per_element, node_state_indices, indices}
    );

    auto indices_host = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_host, indices);
    for (auto row = 0U; row < elem1_num_dof; ++row) {
        for (auto column = 0U; column < elem1_num_dof; ++column) {
            EXPECT_EQ(indices_host(row * elem1_num_dof + column), column);
        }
    }

    for (auto row = 0U; row < elem2_num_dof; ++row) {
        for (auto column = 0U; column < elem2_num_dof; ++column) {
            const auto index = elem1_num_dof * elem1_num_dof + row * elem2_num_dof + column;
            EXPECT_EQ(indices_host(index), column + elem1_num_dof);
        }
    }
}
}  // namespace openturbine::tests
