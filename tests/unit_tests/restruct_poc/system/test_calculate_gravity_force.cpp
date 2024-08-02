#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"

#include "src/restruct_poc/system/calculate_gravity_force.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::tests {

TEST(CalculateGravityForceTests, OneNode) {
    const auto Muu = Kokkos::View<double[1][6][6]>("Muu");
    constexpr auto Muu_data = std::array{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                         13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
                                         25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    const auto Muu_host = Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(Muu_data.data());
    const auto Muu_mirror = Kokkos::create_mirror(Muu);
    Kokkos::deep_copy(Muu_mirror, Muu_host);
    Kokkos::deep_copy(Muu, Muu_mirror);

    const auto eta_tilde = Kokkos::View<double[1][3][3]>("eta_tilde");
    constexpr auto eta_tilde_data = std::array{37., 38., 39., 40., 41., 42., 43., 44., 45.};
    const auto eta_tilde_host =
        Kokkos::View<const double[1][3][3], Kokkos::HostSpace>(eta_tilde_data.data());
    const auto eta_tilde_mirror = Kokkos::create_mirror(eta_tilde);
    Kokkos::deep_copy(eta_tilde_mirror, eta_tilde_host);
    Kokkos::deep_copy(eta_tilde, eta_tilde_mirror);

    const auto gravity = Kokkos::View<double[3]>("gravity");
    constexpr auto gravity_data = std::array{46., 47., 48.};
    const auto gravity_host = Kokkos::View<const double[3], Kokkos::HostSpace>(gravity_data.data());
    const auto gravity_mirror = Kokkos::create_mirror(gravity);
    Kokkos::deep_copy(gravity_mirror, gravity_host);
    Kokkos::deep_copy(gravity, gravity_mirror);

    const auto FG = Kokkos::View<double[1][6]>("FG");

    Kokkos::parallel_for(
        "CalculateGravityForce", 1, CalculateGravityForce{gravity, Muu, eta_tilde, FG}
    );

    constexpr auto FG_exact_data = std::array{46., 47., 48., 5360., 5783., 6206.};
    const auto FG_exact = Kokkos::View<const double[1][6], Kokkos::HostSpace>(FG_exact_data.data());

    const auto FG_mirror = Kokkos::create_mirror(FG);
    Kokkos::deep_copy(FG_mirror, FG);
    CompareWithExpected(FG_mirror, FG_exact);
}

}  // namespace openturbine::tests