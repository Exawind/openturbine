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

    auto node_state_indices = Kokkos::View<int[num_nodes]>("node_state_indices");
    auto node_state_indices_host_data = std::array{0, 1, 2, 3, 4};
    auto node_state_indices_host =
        Kokkos::View<int[num_nodes], Kokkos::HostSpace>(node_state_indices_host_data.data());
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
    constexpr auto elem1_num_nodes = 5;
    constexpr auto elem1_num_dof = elem1_num_nodes * 6;
    constexpr auto elem2_num_nodes = 3;
    constexpr auto elem2_num_dof = elem2_num_nodes * 6;
    constexpr auto num_nodes = elem1_num_nodes + elem2_num_nodes;

    auto elem_indices_host =
        Kokkos::View<Beams::ElemIndices[2], Kokkos::HostSpace>("elem_indices_host");
    auto elem_indices = Kokkos::View<Beams::ElemIndices[2]>("elem_indices");
    elem_indices_host(0).num_nodes = elem1_num_nodes;
    elem_indices_host(0).node_range.first = 0;
    elem_indices_host(1).num_nodes = elem2_num_nodes;
    elem_indices_host(1).node_range.first = elem1_num_nodes;
    Kokkos::deep_copy(elem_indices, elem_indices_host);

    auto node_state_indices = Kokkos::View<int[num_nodes]>("node_state_indices");
    auto node_state_indices_host_data = std::array{0, 1, 2, 3, 4, 5, 6, 7};
    auto node_state_indices_host =
        Kokkos::View<int[num_nodes], Kokkos::HostSpace>(node_state_indices_host_data.data());
    Kokkos::deep_copy(node_state_indices, node_state_indices_host);
    auto indices =
        Kokkos::View<int[elem1_num_dof * elem1_num_dof + elem2_num_dof * elem2_num_dof]>("indices");

    Kokkos::parallel_for(1, PopulateSparseIndices{elem_indices, node_state_indices, indices});

    auto indices_host = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_host, indices);
    for (int row = 0; row < elem1_num_dof; ++row) {
        for (int column = 0; column < elem1_num_dof; ++column) {
            EXPECT_EQ(indices_host(row * elem1_num_dof + column), column);
        }
    }

    auto* indices_host_elem_2 = &indices_host(elem1_num_dof * elem1_num_dof);
    for (int row = 0; row < elem2_num_dof; ++row) {
        for (int column = 0; column < elem2_num_dof; ++column) {
            EXPECT_EQ(indices_host_elem_2[row * elem2_num_dof + column], column + elem1_num_dof);
        }
    }
}
}  // namespace openturbine::restruct_poc::tests