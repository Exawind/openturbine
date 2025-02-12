#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/masses/calculate_mass_matrix_components.hpp"
#include "test_calculate.hpp"

namespace openturbine::tests {

struct ExecuteCalculateMassMatrixComponents {
    Kokkos::View<double[6][6]>::const_type Muu;
    Kokkos::View<double[3]> eta;
    Kokkos::View<double[3][3]> rho;

    KOKKOS_FUNCTION
    void operator()(size_t) const {
        masses::CalculateEta(Muu, eta);
        masses::CalculateRho(Muu, rho);
    }
};

TEST(CalculateMassMatrixComponentsMassesTests, OneQuadPoint) {
    const auto Muu = Kokkos::View<double[6][6]>("Muu");
    constexpr auto Muu_data = std::array{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                         13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
                                         25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    const auto Muu_host = Kokkos::View<const double[6][6], Kokkos::HostSpace>(Muu_data.data());
    const auto Muu_mirror = Kokkos::create_mirror(Muu);
    Kokkos::deep_copy(Muu_mirror, Muu_host);
    Kokkos::deep_copy(Muu, Muu_mirror);

    const auto eta = Kokkos::View<double[3]>("eta");
    const auto rho = Kokkos::View<double[3][3]>("rho");

    Kokkos::parallel_for(
        "CalculateMassMatrixComponents", 1,
        ExecuteCalculateMassMatrixComponents{Muu, eta, rho}
    );

    constexpr auto eta_exact_data = std::array{32., -31., 25.};
    const auto eta_exact =
        Kokkos::View<const double[3], Kokkos::HostSpace>(eta_exact_data.data());

    const auto eta_mirror = Kokkos::create_mirror(eta);
    Kokkos::deep_copy(eta_mirror, eta);
    CompareWithExpected(eta_mirror, eta_exact);

    constexpr auto rho_exact_data = std::array{22., 23., 24., 28., 29., 30., 34., 35., 36.};
    const auto rho_exact =
        Kokkos::View<const double[3][3], Kokkos::HostSpace>(rho_exact_data.data());

    const auto rho_mirror = Kokkos::create_mirror(rho);
    Kokkos::deep_copy(rho_mirror, rho);
    CompareWithExpected(rho_mirror, rho_exact);
}

}  // namespace openturbine::tests
