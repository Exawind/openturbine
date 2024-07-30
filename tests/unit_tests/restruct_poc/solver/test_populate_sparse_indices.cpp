#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/solver/populate_sparse_indices.hpp"

namespace openturbine::restruct_poc::tests {

TEST(PopulateSparseIndices, SingleElement) {
    constexpr auto num_nodes = 5;
    constexpr auto num_dof = num_nodes * 6;

    auto elem_indices_host =
        Kokkos::View<Beams::ElemIndices[1], Kokkos::HostSpace>("elem_indices_host");
    auto elem_indices = Kokkos::View<Beams::ElemIndices[1]>("elem_indices");
    elem_indices_host(0).num_nodes = num_nodes;
    elem_indices_host(0).node_range.first = 0;
    Kokkos::deep_copy(elem_indices, elem_indices_host);

    auto node_state_indices = Kokkos::View<size_t[num_nodes]>("node_state_indices");
    auto node_state_indices_host_data = std::array<size_t, num_nodes>{0U, 1U, 2U, 3U, 4U};
    auto node_state_indices_host =
        Kokkos::View<size_t[num_nodes], Kokkos::HostSpace>(node_state_indices_host_data.data());
    Kokkos::deep_copy(node_state_indices, node_state_indices_host);
    auto indices = Kokkos::View<int[num_dof * num_dof]>("indices");

    Kokkos::parallel_for(1, PopulateSparseIndices{elem_indices, node_state_indices, indices});

    auto indices_host = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_host, indices);
    for (int row = 0; row < num_dof; ++row) {
        for (int column = 0; column < num_dof; ++column) {
            ASSERT_EQ(indices_host(row * num_dof + column), column);
        }
    }
}

TEST(PopulateSparseIndices, TwoElements) {
    constexpr auto elem1_num_nodes = 5U;
    constexpr auto elem1_num_dof = elem1_num_nodes * 6U;
    constexpr auto elem2_num_nodes = 3U;
    constexpr auto elem2_num_dof = elem2_num_nodes * 6U;
    constexpr auto num_nodes = elem1_num_nodes + elem2_num_nodes;

    auto elem_indices_host =
        Kokkos::View<Beams::ElemIndices[2], Kokkos::HostSpace>("elem_indices_host");
    auto elem_indices = Kokkos::View<Beams::ElemIndices[2]>("elem_indices");
    elem_indices_host(0).num_nodes = elem1_num_nodes;
    elem_indices_host(0).node_range.first = 0;
    elem_indices_host(1).num_nodes = elem2_num_nodes;
    elem_indices_host(1).node_range.first = elem1_num_nodes;
    Kokkos::deep_copy(elem_indices, elem_indices_host);

    auto node_state_indices = Kokkos::View<size_t[num_nodes]>("node_state_indices");
    auto node_state_indices_host_data =
        std::array<size_t, num_nodes>{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};
    auto node_state_indices_host =
        Kokkos::View<size_t[num_nodes], Kokkos::HostSpace>(node_state_indices_host_data.data());
    Kokkos::deep_copy(node_state_indices, node_state_indices_host);
    auto indices =
        Kokkos::View<int[elem1_num_dof * elem1_num_dof + elem2_num_dof * elem2_num_dof]>("indices");

    Kokkos::parallel_for(1, PopulateSparseIndices{elem_indices, node_state_indices, indices});

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
}  // namespace openturbine::restruct_poc::tests