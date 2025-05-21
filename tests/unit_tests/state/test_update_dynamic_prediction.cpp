#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "create_view.hpp"
#include "state/update_dynamic_prediction.hpp"

namespace openturbine::tests {

inline void CompareWithExpected(
    const Kokkos::View<const double**>::host_mirror_type& result,
    const Kokkos::View<const double**, Kokkos::HostSpace>& expected
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            EXPECT_NEAR(result(i, j), expected(i, j), 1.e-14);
        }
    }
}

TEST(UpdateDynamicPrediction, OneNode) {
    constexpr auto h = .5;
    constexpr auto beta_prime = 2.;
    constexpr auto gamma_prime = 3.;

    const auto node_freedom_allocation_table =
        CreateView<FreedomSignature[1]>("nfat", std::array{FreedomSignature::AllComponents});
    const auto node_freedom_map_table = CreateView<size_t[1]>("nfmt", std::array{0UL});
    const auto x_delta = CreateLeftView<double[6][1]>("x_delta", std::array{1., 2., 3., 4., 5., 6.});

    const auto q_delta = Kokkos::View<double[1][6]>("q_delta");
    const auto v = Kokkos::View<double[1][6]>("v");
    const auto vd = Kokkos::View<double[1][6]>("vd");

    Kokkos::parallel_for(
        "UpdateDynamicPrediction", 1,
        UpdateDynamicPrediction<Kokkos::DefaultExecutionSpace>{
            h, beta_prime, gamma_prime, node_freedom_allocation_table, node_freedom_map_table,
            x_delta, q_delta, v, vd
        }
    );

    constexpr auto q_delta_exact_data = std::array{2., 4., 6., 8., 10., 12.};
    const auto q_delta_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(q_delta_exact_data.data());
    const auto q_delta_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q_delta);
    CompareWithExpected(q_delta_mirror, q_delta_exact);

    constexpr auto v_exact_data = std::array{3., 6., 9., 12., 15., 18.};
    const auto v_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(v_exact_data.data());
    const auto v_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    CompareWithExpected(v_mirror, v_exact);

    constexpr auto vd_exact_data = std::array{2., 4., 6., 8., 10., 12.};
    const auto vd_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(vd_exact_data.data());
    const auto vd_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vd);
    CompareWithExpected(vd_mirror, vd_exact);
}

}  // namespace openturbine::tests
