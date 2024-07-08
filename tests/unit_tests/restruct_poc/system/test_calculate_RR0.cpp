#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_RR0.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateRR0Tests, OneNode) {
    auto r0 = Kokkos::View<double[1][4]>("r0");
    auto r0_host_data = std::array<double, 4>{1., 2., 3., 4.};
    auto r0_host = Kokkos::View<double[1][4], Kokkos::HostSpace>(r0_host_data.data());
    auto r0_mirror = Kokkos::create_mirror(r0);
    Kokkos::deep_copy(r0_mirror, r0_host);
    Kokkos::deep_copy(r0, r0_mirror);

    auto r = Kokkos::View<double[1][4]>("r");
    auto r_host_data = std::array<double, 4>{5., 6., 7., 8.};
    auto r_host = Kokkos::View<double[1][4], Kokkos::HostSpace>(r_host_data.data());
    auto r_mirror = Kokkos::create_mirror(r);
    Kokkos::deep_copy(r_mirror, r_host);
    Kokkos::deep_copy(r, r_mirror);

    auto rr0 = Kokkos::View<double[1][6][6]>("rr0");

    Kokkos::parallel_for("CalculateRR0", 1, CalculateRR0{r0, r, rr0});

    auto expected_rr0_data = std::array<double, 36>{2780., 4400., -400., 0., 0., 0., 
                                                    -3280., 2372., 3296., 0., 0., 0., 
                                                    2960., -1504., 4028., 0., 0., 0.,
                                                    0., 0., 0., 2780., 4400., -400.,
                                                    0., 0., 0., -3280., 2372., 3296.,
                                                    0., 0., 0., 2960., -1504., 4028.};
    auto expected_rr0 = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(expected_rr0_data.data());

    auto rr0_mirror = Kokkos::create_mirror(rr0);
    Kokkos::deep_copy(rr0_mirror, rr0);
    for(int i = 0; i < 6; ++i) {
        for(int j = 0; j < 6; ++j) {
            EXPECT_EQ(rr0_mirror(0, i, j), expected_rr0(0, i, j));
        }
    }
}

TEST(CalculateRR0Tests, TwoNodes) {
    auto r0 = Kokkos::View<double[2][4]>("r0");
    auto r0_host_data = std::array<double, 8>{1., 2., 3., 4., 5., 6., 7., 8.};
    auto r0_host = Kokkos::View<double[2][4], Kokkos::HostSpace>(r0_host_data.data());
    auto r0_mirror = Kokkos::create_mirror(r0);
    Kokkos::deep_copy(r0_mirror, r0_host);
    Kokkos::deep_copy(r0, r0_mirror);

    auto r = Kokkos::View<double[2][4]>("r");
    auto r_host_data = std::array<double, 8>{5., 6., 7., 8., 9., 10., 11., 12.};
    auto r_host = Kokkos::View<double[2][4], Kokkos::HostSpace>(r_host_data.data());
    auto r_mirror = Kokkos::create_mirror(r);
    Kokkos::deep_copy(r_mirror, r_host);
    Kokkos::deep_copy(r, r_mirror);

    auto rr0 = Kokkos::View<double[2][6][6]>("rr0");

    Kokkos::parallel_for("CalculateRR0", 2, CalculateRR0{r0, r, rr0});

    auto expected_rr0_data = std::array<double, 72>{2780., 4400., -400., 0., 0., 0., 
                                                    -3280., 2372., 3296., 0., 0., 0., 
                                                    2960., -1504., 4028., 0., 0., 0.,
                                                    0., 0., 0., 2780., 4400., -400.,
                                                    0., 0., 0., -3280., 2372., 3296.,
                                                    0., 0., 0., 2960., -1504., 4028.,

                                                    16412., 74896., -11984., 0., 0., 0., 
                                                    -27376., 17284., 70528., 0., 0., 0., 
                                                    70736., -10688., 30076., 0., 0., 0.,
                                                    0., 0., 0., 16412., 74896., -11984.,
                                                    0., 0., 0., -27376., 17284., 70528.,
                                                    0., 0., 0., 70736., -10688., 30076.};
    auto expected_rr0 = Kokkos::View<double[2][6][6], Kokkos::HostSpace>(expected_rr0_data.data());

    auto rr0_mirror = Kokkos::create_mirror(rr0);
    Kokkos::deep_copy(rr0_mirror, rr0);
    for(int i = 0; i < 6; ++i) {
        for(int j = 0; j < 6; ++j) {
            EXPECT_EQ(rr0_mirror(0, i, j), expected_rr0(0, i, j));
        }
    }
    for(int i = 0; i < 6; ++i) {
        for(int j = 0; j < 6; ++j) {
            EXPECT_EQ(rr0_mirror(1, i, j), expected_rr0(1, i, j));
        }
    }
}

}  // namespace openturbine::restruct_poc::tests