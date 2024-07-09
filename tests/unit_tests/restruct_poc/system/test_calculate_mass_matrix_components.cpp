#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_mass_matrix_components.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateMassMatrixComponentsTests, OneQuadPoint) {
    auto Muu = Kokkos::View<double[1][6][6]>("Muu");
    auto Muu_data = std::array<double, 36>{ 1.,  2.,  3.,  4.,  5.,  6.,
                                            7.,  8.,  9., 10., 11., 12.,
                                           13., 14., 15., 16., 17., 18.,
                                           19., 20., 21., 22., 23., 24.,
                                           25., 26., 27., 28., 29., 30.,
                                           31., 32., 33., 34., 35., 36.};
    auto Muu_host = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(Muu_data.data());
    auto Muu_mirror = Kokkos::create_mirror(Muu);
    Kokkos::deep_copy(Muu_mirror, Muu_host);
    Kokkos::deep_copy(Muu, Muu_mirror);

    auto eta = Kokkos::View<double[1][3]>("eta");
    auto rho = Kokkos::View<double[1][3][3]>("rho");
    auto eta_tilde = Kokkos::View<double[1][3][3]>("eta_tilde");

    Kokkos::parallel_for("CalculateMassMatrixComponents", 1, CalculateMassMatrixComponents{Muu, eta, rho, eta_tilde});

    auto eta_exact_data = std::array<double, 3>{32., -31., 25.};
    auto eta_exact = Kokkos::View<double[1][3], Kokkos::HostSpace>(eta_exact_data.data());

    auto eta_mirror = Kokkos::create_mirror(eta);
    Kokkos::deep_copy(eta_mirror, eta);
    for(int i = 0; i < 3; ++i) {
        EXPECT_EQ(eta_mirror(0, i), eta_exact(0, i));
    }

    auto rho_exact_data = std::array<double, 9>{22., 23., 24.,
                                                28., 29., 30.,
                                                34., 35., 36.};
    auto rho_exact = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(rho_exact_data.data());

    auto rho_mirror = Kokkos::create_mirror(rho);
    Kokkos::deep_copy(rho_mirror, rho);
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            EXPECT_EQ(rho_mirror(0, i, j), rho_exact(0, i, j));
        }
    }

    auto eta_tilde_exact_data = std::array<double, 9>{ 0., -25., -31.,
                                                      25.,   0., -32.,
                                                      31.,  32.,   0.};
    auto eta_tilde_exact = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(eta_tilde_exact_data.data());

    auto eta_tilde_mirror = Kokkos::create_mirror(eta_tilde);
    Kokkos::deep_copy(eta_tilde_mirror, eta_tilde);
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            EXPECT_EQ(eta_tilde_mirror(0, i, j), eta_tilde_exact(0, i, j));
        }
    }
}

}  // namespace openturbine::restruct_poc::tests