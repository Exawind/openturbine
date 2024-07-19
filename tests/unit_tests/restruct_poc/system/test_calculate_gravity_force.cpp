#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_gravity_force.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateGravityForceTests, OneNode) {
    auto Muu = Kokkos::View<double[1][6][6]>("Muu");
    auto Muu_data = std::array<double, 36>{
        1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14., 15., 16., 17., 18.,
        19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    auto Muu_host = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(Muu_data.data());
    auto Muu_mirror = Kokkos::create_mirror(Muu);
    Kokkos::deep_copy(Muu_mirror, Muu_host);
    Kokkos::deep_copy(Muu, Muu_mirror);

    auto eta_tilde = Kokkos::View<double[1][3][3]>("eta_tilde");
    auto eta_tilde_data = std::array<double, 9>{37., 38., 39., 40., 41., 42., 43., 44., 45.};
    auto eta_tilde_host = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(eta_tilde_data.data());
    auto eta_tilde_mirror = Kokkos::create_mirror(eta_tilde);
    Kokkos::deep_copy(eta_tilde_mirror, eta_tilde_host);
    Kokkos::deep_copy(eta_tilde, eta_tilde_mirror);

    auto gravity = Kokkos::View<double[3]>("gravity");
    auto gravity_data = std::array<double, 3>{46., 47., 48.};
    auto gravity_host = Kokkos::View<double[3], Kokkos::HostSpace>(gravity_data.data());
    auto gravity_mirror = Kokkos::create_mirror(gravity);
    Kokkos::deep_copy(gravity_mirror, gravity_host);
    Kokkos::deep_copy(gravity, gravity_mirror);

    auto FG = Kokkos::View<double[1][6]>("FG");

    Kokkos::parallel_for(
        "CalculateGravityForce", 1, CalculateGravityForce{gravity, Muu, eta_tilde, FG}
    );

    auto FG_exact_data = std::array<double, 6>{46., 47., 48., 5360., 5783., 6206.};
    auto FG_exact = Kokkos::View<double[1][6], Kokkos::HostSpace>(FG_exact_data.data());

    auto FG_mirror = Kokkos::create_mirror(FG);
    Kokkos::deep_copy(FG_mirror, FG);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(FG_mirror(0, i), FG_exact(0, i));
    }
}

}  // namespace openturbine::restruct_poc::tests