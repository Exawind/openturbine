#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/solver/fill_unshifted_row_ptrs.hpp"

namespace openturbine::tests {

TEST(FillUnshiftedRowPtrs, Shift1By1) {
    constexpr auto num_initial_rows = 1U;
    constexpr auto num_final_rows = 2U;

    constexpr auto old_row_ptrs_host_data = std::array<size_t, num_initial_rows + 1U>{0U, 5U};
    const auto old_row_ptrs_host =
        Kokkos::View<const size_t[num_initial_rows + 1U], Kokkos::HostSpace>(
            old_row_ptrs_host_data.data()
        );
    const auto old_row_ptrs = Kokkos::View<size_t[num_initial_rows + 1U]>("old_row_ptrs");
    Kokkos::deep_copy(old_row_ptrs, old_row_ptrs_host);

    const auto new_row_ptrs = Kokkos::View<size_t[num_final_rows + 1U]>("new_row_ptrs");

    Kokkos::parallel_for(
        "FillUnshiftedRowPtrs", num_final_rows + 1U,
        FillUnshiftedRowPtrs<Kokkos::View<size_t*>>{num_initial_rows, old_row_ptrs, new_row_ptrs}
    );

    constexpr auto new_row_ptrs_exact_host = std::array<size_t, num_final_rows + 1U>{0U, 5U, 5U};
    const auto new_row_ptrs_exact =
        Kokkos::View<const size_t[num_final_rows + 1U], Kokkos::HostSpace>(
            new_row_ptrs_exact_host.data()
        );

    const auto new_row_ptrs_host = Kokkos::create_mirror(new_row_ptrs);
    Kokkos::deep_copy(new_row_ptrs_host, new_row_ptrs);

    for (auto i = 0U; i < num_final_rows + 1U; ++i) {
        EXPECT_EQ(new_row_ptrs_host(i), new_row_ptrs_exact(i));
    }
}

}  // namespace openturbine::tests
