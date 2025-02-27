#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/calculate_Ouu.hpp"
#include "test_calculate.hpp"

namespace {

void TestCalculateOuu() {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    constexpr auto Cuu_data = std::array{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                         13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
                                         25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    const auto Cuu_host = Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type(Cuu_data.data());
    const auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu_host);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");
    constexpr auto x0pupSS_data = std::array{37., 38., 39., 40., 41., 42., 43., 44., 45.};
    const auto x0pupSS_host =
        Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type(x0pupSS_data.data());
    const auto x0pupSS_mirror = Kokkos::create_mirror(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, x0pupSS_host);
    Kokkos::deep_copy(x0pupSS, x0pupSS_mirror);

    const auto M_tilde = Kokkos::View<double[3][3]>("M_tilde");
    constexpr auto M_tilde_data = std::array{46., 47., 48., 49., 50., 51., 52., 53., 54.};
    const auto M_tilde_host =
        Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type(M_tilde_data.data());
    const auto M_tilde_mirror = Kokkos::create_mirror(M_tilde);
    Kokkos::deep_copy(M_tilde_mirror, M_tilde_host);
    Kokkos::deep_copy(M_tilde, M_tilde_mirror);

    const auto N_tilde = Kokkos::View<double[3][3]>("N_tilde");
    constexpr auto N_tilde_data = std::array{55., 56., 57., 58., 59., 60., 61., 62., 63.};
    const auto N_tilde_host =
        Kokkos::View<const double[3][3], Kokkos::HostSpace>::const_type(N_tilde_data.data());
    const auto N_tilde_mirror = Kokkos::create_mirror(N_tilde);
    Kokkos::deep_copy(N_tilde_mirror, N_tilde_host);
    Kokkos::deep_copy(N_tilde, N_tilde_mirror);

    const auto Ouu = Kokkos::View<double[6][6]>("Ouu");

    Kokkos::parallel_for(
        "CalculateOuu", 1,
        KOKKOS_LAMBDA(size_t) {
            openturbine::beams::CalculateOuu(Cuu, x0pupSS, M_tilde, N_tilde, Ouu);
        }
    );

    constexpr auto Ouu_exact_data =
        std::array{0., 0., 0., 191.,  196.,  201.,  0., 0., 0., 908.,  931.,  954.,
                   0., 0., 0., 1625., 1666., 1707., 0., 0., 0., 2360., 2419., 2478.,
                   0., 0., 0., 3077., 3154., 3231., 0., 0., 0., 3794., 3889., 3984.};
    const auto Ouu_exact =
        Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type(Ouu_exact_data.data());

    const auto Ouu_mirror = Kokkos::create_mirror(Ouu);
    Kokkos::deep_copy(Ouu_mirror, Ouu);
    openturbine::tests::CompareWithExpected(Ouu_mirror, Ouu_exact);
}

}  // namespace

namespace openturbine::tests {

TEST(CalculateOuuTests, OneNode) {
    TestCalculateOuu();
}

}  // namespace openturbine::tests
