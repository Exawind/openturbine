#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_force_FC.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateForceFCTests, OneNode) {
    auto Cuu = Kokkos::View<double[1][6][6]>("Cuu");
    auto Cuu_data = std::array<double, 36>{ 1.,  2.,  3.,  4.,  5.,  6.,
                                            7.,  8.,  9., 10., 11., 12.,
                                           13., 14., 15., 16., 17., 18.,
                                           19., 20., 21., 22., 23., 24.,
                                           25., 26., 27., 28., 29., 30.,
                                           31., 32., 33., 34., 35., 36.,};
    auto Cuu_host = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(Cuu_data.data());
    auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu_host);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    auto strain = Kokkos::View<double[1][6]>("strain");
    auto strain_data = std::array<double, 6>{37., 38., 39., 40., 41., 42.};
    auto strain_host = Kokkos::View<double[1][6], Kokkos::HostSpace>(strain_data.data());
    auto strain_mirror = Kokkos::create_mirror(strain);
    Kokkos::deep_copy(strain_mirror, strain_host);
    Kokkos::deep_copy(strain, strain_mirror);

    auto FC = Kokkos::View<double[1][6]>("FC");
    auto M_tilde = Kokkos::View<double[1][3][3]>("M_tilde");
    auto N_tilde = Kokkos::View<double[1][3][3]>("N_tilde");

    Kokkos::parallel_for("CalculateForceFC", 1, CalculateForceFC{Cuu, strain, FC, M_tilde, N_tilde});

    auto FC_exact_data = std::array<double, 6>{847., 2269., 3691., 5113., 6535., 7957.};
    auto FC_exact = Kokkos::View<double[1][6], Kokkos::HostSpace>(FC_exact_data.data());

    auto FC_mirror = Kokkos::create_mirror(FC);
    Kokkos::deep_copy(FC_mirror, FC);
    for(int i = 0; i < 6; ++i) {
        EXPECT_EQ(FC_mirror(0, i), FC_exact(0, i));
    }

    auto M_tilde_exact_data = std::array<double, 9>{0., -7957., 6535., 
                                                    7957., 0., -5113., 
                                                    -6535., 5113., 0.};
    auto M_tilde_exact = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(M_tilde_exact_data.data());

    auto M_tilde_mirror = Kokkos::create_mirror(M_tilde);
    Kokkos::deep_copy(M_tilde_mirror, M_tilde);
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            EXPECT_EQ(M_tilde_mirror(0, i, j), M_tilde_exact(0, i, j));
        }
    }

    auto N_tilde_exact_data = std::array<double, 9>{0., -3691., 2269., 
                                                    3691., 0., -847.,
                                                    -2269., 847., 0.};
    auto N_tilde_exact = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(N_tilde_exact_data.data());

    auto N_tilde_mirror = Kokkos::create_mirror(N_tilde);
    Kokkos::deep_copy(N_tilde_mirror, N_tilde);
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            EXPECT_EQ(N_tilde_mirror(0, i, j), N_tilde_exact(0, i, j));
        }
    }
}

}  // namespace openturbine::restruct_poc::tests