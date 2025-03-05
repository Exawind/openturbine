#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

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

    constexpr auto node_freedom_allocation_table_host_data =
        std::array<FreedomSignature, 1>{FreedomSignature::AllComponents};
    const auto node_freedom_allocation_table_host =
        Kokkos::View<FreedomSignature[1], Kokkos::HostSpace>::const_type(
            node_freedom_allocation_table_host_data.data()
        );
    const auto node_freedom_allocation_table = Kokkos::View<FreedomSignature[1]>("nfat");
    const auto node_freedom_allocation_table_mirror =
        Kokkos::create_mirror(node_freedom_allocation_table);
    Kokkos::deep_copy(node_freedom_allocation_table_mirror, node_freedom_allocation_table_host);
    Kokkos::deep_copy(node_freedom_allocation_table, node_freedom_allocation_table_mirror);

    constexpr auto node_freedom_map_table_host_data = std::array<size_t, 1>{0UL};
    const auto node_freedom_map_table_host = Kokkos::View<size_t[1], Kokkos::HostSpace>::const_type(
        node_freedom_map_table_host_data.data()
    );
    const auto node_freedom_map_table = Kokkos::View<size_t[1]>("nfmt");
    const auto node_freedom_map_table_mirror = Kokkos::create_mirror(node_freedom_map_table);
    Kokkos::deep_copy(node_freedom_map_table_mirror, node_freedom_map_table_host);
    Kokkos::deep_copy(node_freedom_map_table, node_freedom_map_table_mirror);

    constexpr auto x_delta_host_data = std::array{1., 2., 3., 4., 5., 6.};
    const auto x_delta_host =
        Kokkos::View<double[6][1], Kokkos::HostSpace>::const_type(x_delta_host_data.data());
    const auto x_delta = Kokkos::View<double[6][1], Kokkos::LayoutLeft>("x_delta");
    const auto x_delta_mirror = Kokkos::create_mirror(x_delta);
    Kokkos::deep_copy(x_delta_mirror, x_delta_host);
    Kokkos::deep_copy(x_delta, x_delta_mirror);

    const auto q_delta = Kokkos::View<double[1][6]>("q_delta");
    const auto v = Kokkos::View<double[1][6]>("v");
    const auto vd = Kokkos::View<double[1][6]>("vd");

    Kokkos::parallel_for(
        "UpdateDynamicPrediction", 1,
        UpdateDynamicPrediction{
            h, beta_prime, gamma_prime, node_freedom_allocation_table, node_freedom_map_table,
            x_delta, q_delta, v, vd
        }
    );

    constexpr auto q_delta_exact_data = std::array{2., 4., 6., 8., 10., 12.};
    const auto q_delta_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(q_delta_exact_data.data());
    const auto q_delta_mirror = Kokkos::create_mirror(q_delta);
    Kokkos::deep_copy(q_delta_mirror, q_delta);
    CompareWithExpected(q_delta_mirror, q_delta_exact);

    constexpr auto v_exact_data = std::array{3., 6., 9., 12., 15., 18.};
    const auto v_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(v_exact_data.data());
    const auto v_mirror = Kokkos::create_mirror(v);
    Kokkos::deep_copy(v_mirror, v);
    CompareWithExpected(v_mirror, v_exact);

    constexpr auto vd_exact_data = std::array{2., 4., 6., 8., 10., 12.};
    const auto vd_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(vd_exact_data.data());
    const auto vd_mirror = Kokkos::create_mirror(vd);
    Kokkos::deep_copy(vd_mirror, vd);
    CompareWithExpected(vd_mirror, vd_exact);
}

}  // namespace openturbine::tests
