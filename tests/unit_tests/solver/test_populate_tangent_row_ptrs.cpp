#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/populate_tangent_row_ptrs.hpp"

namespace openturbine::tests {

TEST(PopulateTangentRowPtrs, TwoNodeSystem) {
    constexpr auto num_system_nodes = 2U;
    constexpr auto num_system_dofs = 6U * num_system_nodes;
    constexpr auto num_entries = num_system_dofs + 1U;
    const auto row_ptrs = Kokkos::View<size_t[num_entries]>("row_ptrs");
    Kokkos::parallel_for(
        "PopulateTangentRowPtrs", 1, PopulateTangentRowPtrs<size_t>{num_system_nodes, row_ptrs}
    );

    constexpr auto exact_row_ptrs_data =
        std::array{0U, 6U, 12U, 18U, 24U, 30U, 36U, 42U, 48U, 54U, 60U, 66U, 72U, 78U};
    const auto exact_row_ptrs =
        Kokkos::View<const unsigned[num_entries], Kokkos::HostSpace>(exact_row_ptrs_data.data());
    const auto row_ptrs_host = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_host, row_ptrs);
    for (auto i = 0U; i < num_entries; ++i) {
        EXPECT_EQ(row_ptrs_host(i), exact_row_ptrs(i));
    }
}

}  // namespace openturbine::tests
