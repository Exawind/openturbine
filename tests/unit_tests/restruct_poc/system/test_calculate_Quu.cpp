#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_Quu.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateQuuTests, OneNode) {
    auto Cuu = Kokkos::View<double[1][6][6]>("Cuu");
    auto Cuu_data = std::array<double, 36>{1., 2., 3., 4., 5., 6., 
                                           7., 8., 9., 10., 11., 12., 
                                           13., 14., 15., 16., 17., 18., 
                                           19., 20., 21., 22., 23., 24., 
                                           25., 26., 27., 28., 29., 30., 
                                           31., 32., 33., 34., 35., 36.};
    auto Cuu_host = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(Cuu_data.data());
    auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu_host);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    auto x0pupSS = Kokkos::View<double[1][3][3]>("x0pupSS");
    auto x0pupSS_data = std::array<double, 9>{37., 38., 39., 40., 41., 42., 43., 44., 45.};
    auto x0pupSS_host = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(x0pupSS_data.data());
    auto x0pupSS_mirror = Kokkos::create_mirror(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, x0pupSS_host);
    Kokkos::deep_copy(x0pupSS, x0pupSS_mirror);

    auto M_tilde = Kokkos::View<double[1][3][3]>("M_tilde");
    auto M_tilde_data = std::array<double, 9>{46., 47., 48., 49., 50., 51., 52., 53., 54.};
    auto M_tilde_host = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(M_tilde_data.data());
    auto M_tilde_mirror = Kokkos::create_mirror(M_tilde);
    Kokkos::deep_copy(M_tilde_mirror, M_tilde_host);
    Kokkos::deep_copy(M_tilde, M_tilde_mirror);

    auto N_tilde = Kokkos::View<double[1][3][3]>("N_tilde");
    auto N_tilde_data = std::array<double, 9>{55., 56., 57., 58., 59., 60., 61., 62., 63.};
    auto N_tilde_host = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(N_tilde_data.data());
    auto N_tilde_mirror = Kokkos::create_mirror(N_tilde);
    Kokkos::deep_copy(N_tilde_mirror, N_tilde_host);
    Kokkos::deep_copy(N_tilde, N_tilde_mirror);

    auto Quu = Kokkos::View<double[1][6][6]>("Quu");

    Kokkos::parallel_for("CalculateQuu", 1, CalculateQuu{Cuu, x0pupSS, N_tilde, Quu});

    auto Quu_exact_data = std::array<double, 36>{0., 0., 0., 0., 0., 0., 
                                                 0., 0., 0., 0., 0., 0., 
                                                 0., 0., 0., 0., 0., 0., 
                                                 0., 0., 0., 113262., 116130., 118998.,  
                                                 0., 0., 0., 115986., 118923., 121860.,
                                                 0., 0., 0., 118710., 121716., 124722.};
    auto Quu_exact = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(Quu_exact_data.data());

    auto Quu_mirror = Kokkos::create_mirror(Quu);
    Kokkos::deep_copy(Quu_mirror, Quu);
    for(int i = 0; i < 6; ++i) {
        for(int j = 0; j < 6; ++j) {
            EXPECT_EQ(Quu_mirror(0, i, j), Quu_exact(0, i, j));
        }
    }
}

}  // namespace openturbine::restruct_poc::tests