#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"

#include "system/beams/calculate_RR0.hpp"

namespace openturbine::tests {

TEST(CalculateRR0Tests, OneNode) {
    const auto x = Kokkos::View<double[1][1][7]>("x");
    constexpr auto x_host_data = std::array{1., 2., 3., 4., 5., 6., 7.};
    const auto x_host = Kokkos::View<const double[1][1][7], Kokkos::HostSpace>(x_host_data.data());
    const auto x_mirror = Kokkos::create_mirror(x);
    Kokkos::deep_copy(x_mirror, x_host);
    Kokkos::deep_copy(x, x_mirror);

    const auto rr0 = Kokkos::View<double[1][1][6][6]>("rr0");

    Kokkos::parallel_for("CalculateRR0", 1, CalculateRR0{0, x, rr0});

    constexpr auto expected_rr0_data = std::array{-44., 4.,   118., 0.,   0.,   0.,    //
                                                  116., -22., 44.,  0.,   0.,   0.,    //
                                                  22.,  124., 4.,   0.,   0.,   0.,    //
                                                  0.,   0.,   0.,   -44., 4.,   118.,  //
                                                  0.,   0.,   0.,   116., -22., 44.,   //
                                                  0.,   0.,   0.,   22.,  124., 4.};
    const auto expected_rr0 =
        Kokkos::View<const double[1][1][6][6], Kokkos::HostSpace>(expected_rr0_data.data());

    const auto rr0_mirror = Kokkos::create_mirror(rr0);
    Kokkos::deep_copy(rr0_mirror, rr0);
    CompareWithExpected(rr0_mirror, expected_rr0);
}

TEST(CalculateRR0Tests, TwoNodes) {
    const auto x = Kokkos::View<double[1][2][7]>("x");
    constexpr auto x_host_data =
        std::array{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.};
    const auto x_host = Kokkos::View<const double[1][2][7], Kokkos::HostSpace>(x_host_data.data());
    const auto x_mirror = Kokkos::create_mirror(x);
    Kokkos::deep_copy(x_mirror, x_host);
    Kokkos::deep_copy(x, x_mirror);

    const auto rr0 = Kokkos::View<double[1][2][6][6]>("rr0");

    Kokkos::parallel_for("CalculateRR0", 2, CalculateRR0{0, x, rr0});

    constexpr auto expected_rr0_data = std::array{-44.,  4.,   118., 0.,    0.,   0.,    //
                                                  116.,  -22., 44.,  0.,    0.,   0.,    //
                                                  22.,   124., 4.,   0.,    0.,   0.,    //
                                                  0.,    0.,   0.,   -44.,  4.,   118.,  //
                                                  0.,    0.,   0.,   116.,  -22., 44.,   //
                                                  0.,    0.,   0.,   22.,   124., 4.,

                                                  -100., 4.,   622., 0.,    0.,   0.,    //
                                                  620.,  -50., 100., 0.,    0.,   0.,    //
                                                  50.,   628., 4.,   0.,    0.,   0.,    //
                                                  0.,    0.,   0.,   -100., 4.,   622.,  //
                                                  0.,    0.,   0.,   620.,  -50., 100.,  //
                                                  0.,    0.,   0.,   50.,   628., 4.};
    const auto expected_rr0 =
        Kokkos::View<const double[1][2][6][6], Kokkos::HostSpace>(expected_rr0_data.data());

    const auto rr0_mirror = Kokkos::create_mirror(rr0);
    Kokkos::deep_copy(rr0_mirror, rr0);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_EQ(rr0_mirror(0, 0, i, j), expected_rr0(0, 0, i, j));
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_EQ(rr0_mirror(0, 1, i, j), expected_rr0(0, 1, i, j));
        }
    }
}

}  // namespace openturbine::tests
