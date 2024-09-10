#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/compute_number_of_non_zeros.hpp"

namespace openturbine::tests {

TEST(ComputeNumberOfNonZeros, SingleElement) {
    auto num_nodes = Kokkos::View<size_t[1]>("num_nodes");
    auto num_nodes_host = Kokkos::create_mirror(num_nodes);
    num_nodes_host(0) = size_t{5U};
    Kokkos::deep_copy(num_nodes, num_nodes_host);

    auto num_non_zero = size_t{0U};
    Kokkos::parallel_reduce(1, ComputeNumberOfNonZeros{num_nodes}, num_non_zero);
    constexpr auto num_dof = 5 * 6;
    constexpr auto expected_num_non_zero = num_dof * num_dof;
    EXPECT_EQ(num_non_zero, expected_num_non_zero);
}

TEST(ComputeNumberOfNonZeros, TwoElements) {
    auto num_nodes = Kokkos::View<size_t[2]>("num_nodes");
    auto num_nodes_host = Kokkos::create_mirror(num_nodes);
    num_nodes_host(0) = size_t{5U};
    num_nodes_host(1) = size_t{3U};
    Kokkos::deep_copy(num_nodes, num_nodes_host);

    auto num_non_zero = size_t{0U};
    Kokkos::parallel_reduce(2, ComputeNumberOfNonZeros{num_nodes}, num_non_zero);
    constexpr auto num_dof_elem1 = 5U * 6U;
    constexpr auto num_dof_elem2 = 3U * 6U;
    constexpr auto expected_num_non_zero =
        num_dof_elem1 * num_dof_elem1 + num_dof_elem2 * num_dof_elem2;
    EXPECT_EQ(num_non_zero, expected_num_non_zero);
}
}  // namespace openturbine::tests
