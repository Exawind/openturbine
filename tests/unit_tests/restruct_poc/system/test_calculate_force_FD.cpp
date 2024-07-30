#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"

#include "src/restruct_poc/system/calculate_force_FD.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateForceFDTests, OneNode) {
    const auto x0pupSS = Kokkos::View<double[1][3][3]>("x0pupSS");
    constexpr auto x0pupSS_data = std::array{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    const auto x0pupSS_host =
        Kokkos::View<const double[1][3][3], Kokkos::HostSpace>(x0pupSS_data.data());
    const auto x0pupSS_mirror = Kokkos::create_mirror(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, x0pupSS_host);
    Kokkos::deep_copy(x0pupSS, x0pupSS_mirror);

    const auto FC = Kokkos::View<double[1][6]>("FC");
    constexpr auto FC_data = std::array{10., 11., 12., 13., 14., 15.};
    const auto FC_host = Kokkos::View<const double[1][6], Kokkos::HostSpace>(FC_data.data());
    const auto FC_mirror = Kokkos::create_mirror(FC);
    Kokkos::deep_copy(FC_mirror, FC_host);
    Kokkos::deep_copy(FC, FC_mirror);

    const auto FD = Kokkos::View<double[1][6]>("FD");

    Kokkos::parallel_for("CalculateForceFD", 1, CalculateForceFD{x0pupSS, FC, FD});

    constexpr auto FD_exact_data = std::array{0., 0., 0., 138., 171., 204.};
    const auto FD_exact = Kokkos::View<const double[1][6], Kokkos::HostSpace>(FD_exact_data.data());

    const auto FD_mirror = Kokkos::create_mirror(FD);
    Kokkos::deep_copy(FD_mirror, FD);
    CompareWithExpected(FD_mirror, FD_exact);
}

}  // namespace openturbine::restruct_poc::tests