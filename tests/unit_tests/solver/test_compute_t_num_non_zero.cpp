#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "solver/compute_t_num_non_zero.hpp"

namespace openturbine::tests {
TEST(ComputeTNumNonZero, OneNode) {
    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[1]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto num_non_zero = ComputeTNumNonZero(node_freedom_allocation_table);

    EXPECT_EQ(num_non_zero, 36UL);
}

TEST(ComputeTNumNonZero, TwoNodes) {
    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[2]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto num_non_zero = ComputeTNumNonZero(node_freedom_allocation_table);

    EXPECT_EQ(num_non_zero, 72UL);
}

}  // namespace openturbine::tests
