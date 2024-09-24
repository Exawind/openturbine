#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/state/calculate_next_state.hpp"

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

TEST(CalculateNextState, OneNode) {
    constexpr auto h = 2.;
    constexpr auto alpha_f = 3.;
    constexpr auto alpha_m = 4.;
    constexpr auto beta = 5.;
    constexpr auto gamma = 6.;

    const auto q_delta = Kokkos::View<double[1][6]>("q_delta");
    const auto q_delta_host_data = std::array{1., 2., 3., 4., 5., 6.};
    const auto q_delta_host =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(q_delta_host_data.data());
    const auto q_delta_mirror = Kokkos::create_mirror(q_delta);
    Kokkos::deep_copy(q_delta_mirror, q_delta_host);
    Kokkos::deep_copy(q_delta, q_delta_mirror);

    const auto v = Kokkos::View<double[1][6]>("v");
    constexpr auto v_host_data = std::array{1., 2., 3., 4., 5., 6.};
    const auto v_host =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(v_host_data.data());
    const auto v_mirror = Kokkos::create_mirror(v);
    Kokkos::deep_copy(v_mirror, v_host);
    Kokkos::deep_copy(v, v_mirror);

    const auto vd = Kokkos::View<double[1][6]>("vd");
    constexpr auto vd_host_data = std::array{7., 8., 9., 10., 11., 12.};
    const auto vd_host =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(vd_host_data.data());
    const auto vd_mirror = Kokkos::create_mirror(vd);
    Kokkos::deep_copy(vd_mirror, vd_host);
    Kokkos::deep_copy(vd, vd_mirror);

    const auto a = Kokkos::View<double[1][6]>("a");
    constexpr auto a_host_data = std::array{13., 14., 15., 16., 17., 18.};
    const auto a_host =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(a_host_data.data());
    const auto a_mirror = Kokkos::create_mirror(a);
    Kokkos::deep_copy(a_mirror, a_host);
    Kokkos::deep_copy(a, a_mirror);

    Kokkos::parallel_for(
        "CalculateNextState", 1,
        CalculateNextState{h, alpha_f, alpha_m, beta, gamma, q_delta, v, vd, a}
    );

    constexpr auto q_delta_exact_data = std::array{-12.666666666666661, -17.333333333333339, -22.,
                                                   -26.666666666666661, -31.333333333333339, -36.};
    const auto q_delta_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(q_delta_exact_data.data());
    Kokkos::deep_copy(q_delta_mirror, q_delta);
    CompareWithExpected(q_delta_mirror, q_delta_exact);

    constexpr auto v_exact_data = std::array{-4.9999999999999929, -10.000000000000007, -15.,
                                             -19.999999999999993, -25.000000000000007, -30.};
    const auto v_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(v_exact_data.data());
    Kokkos::deep_copy(v_mirror, v);
    CompareWithExpected(v_mirror, v_exact);

    constexpr auto vd_exact_data = std::array{0., 0., 0., 0., 0., 0.};
    const auto vd_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(vd_exact_data.data());
    Kokkos::deep_copy(vd_mirror, vd);
    CompareWithExpected(vd_mirror, vd_exact);

    constexpr auto a_exact_data = std::array{10.333333333333334, 10.666666666666666, 11.,
                                             11.333333333333334, 11.666666666666666, 12.};
    const auto a_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(a_exact_data.data());
    Kokkos::deep_copy(a_mirror, a);
    CompareWithExpected(a_mirror, a_exact);
}
}  // namespace openturbine::tests
