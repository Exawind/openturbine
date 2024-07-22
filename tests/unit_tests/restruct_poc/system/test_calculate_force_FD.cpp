#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_force_FD.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateForceFDTests, OneNode) {
    auto x0pupSS = Kokkos::View<double[1][3][3]>("x0pupSS");
    auto x0pupSS_data = std::array<double, 9>{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    auto x0pupSS_host = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(x0pupSS_data.data());
    auto x0pupSS_mirror = Kokkos::create_mirror(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, x0pupSS_host);
    Kokkos::deep_copy(x0pupSS, x0pupSS_mirror);

    auto FC = Kokkos::View<double[1][6]>("FC");
    auto FC_data = std::array<double, 6>{10., 11., 12., 13., 14., 15.};
    auto FC_host = Kokkos::View<double[1][6], Kokkos::HostSpace>(FC_data.data());
    auto FC_mirror = Kokkos::create_mirror(FC);
    Kokkos::deep_copy(FC_mirror, FC_host);
    Kokkos::deep_copy(FC, FC_mirror);

    auto FD = Kokkos::View<double[1][6]>("FD");

    Kokkos::parallel_for("CalculateForceFD", 1, CalculateForceFD{x0pupSS, FC, FD});

    auto FD_exact_data = std::array<double, 6>{0., 0., 0., 138., 171., 204.};
    auto FD_exact = Kokkos::View<double[1][6], Kokkos::HostSpace>(FD_exact_data.data());

    auto FD_mirror = Kokkos::create_mirror(FD);
    Kokkos::deep_copy(FD_mirror, FD);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(FD_mirror(0, i), FD_exact(0, i));
    }
}

}  // namespace openturbine::restruct_poc::tests