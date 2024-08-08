#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/solver/compute_number_of_non_zeros.hpp"

namespace openturbine::tests {

TEST(ComputeNumberOfNonZeros, SingleElement) {
    auto elem_indices_host =
        Kokkos::View<Beams::ElemIndices[1], Kokkos::HostSpace>("elem_indices_host");
    auto elem_indices = Kokkos::View<Beams::ElemIndices[1]>("elem_indices");

    elem_indices_host(0).num_nodes = 5;
    Kokkos::deep_copy(elem_indices, elem_indices_host);
    auto num_non_zero = size_t{0U};
    Kokkos::parallel_reduce(1, ComputeNumberOfNonZeros{elem_indices}, num_non_zero);
    constexpr auto num_dof = 5 * 6;
    constexpr auto expected_num_non_zero = num_dof * num_dof;
    EXPECT_EQ(num_non_zero, expected_num_non_zero);
}

TEST(ComputeNumberOfNonZeros, TwoElements) {
    auto elem_indices_host =
        Kokkos::View<Beams::ElemIndices[2], Kokkos::HostSpace>("elem_indices_host");
    auto elem_indices = Kokkos::View<Beams::ElemIndices[2]>("elem_indices");

    elem_indices_host(0).num_nodes = 5;
    elem_indices_host(1).num_nodes = 3;
    Kokkos::deep_copy(elem_indices, elem_indices_host);
    auto num_non_zero = size_t{0U};
    Kokkos::parallel_reduce(2, ComputeNumberOfNonZeros{elem_indices}, num_non_zero);
    constexpr auto num_dof_elem1 = 5 * 6;
    constexpr auto num_dof_elem2 = 3 * 6;
    constexpr auto expected_num_non_zero =
        num_dof_elem1 * num_dof_elem1 + num_dof_elem2 * num_dof_elem2;
    EXPECT_EQ(num_non_zero, expected_num_non_zero);
}
}  // namespace openturbine::tests