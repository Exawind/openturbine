#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/calculate_force_FC.hpp"
#include "test_calculate.hpp"

namespace {

void TestCalculateForceFC() {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    constexpr auto Cuu_data = std::array{
        1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14., 15., 16., 17., 18.,
        19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.,
    };
    const auto Cuu_host = Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type(Cuu_data.data());
    const auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu_host);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    const auto strain = Kokkos::View<double[6]>("strain");
    constexpr auto strain_data = std::array{37., 38., 39., 40., 41., 42.};
    const auto strain_host =
        Kokkos::View<double[6], Kokkos::HostSpace>::const_type(strain_data.data());
    const auto strain_mirror = Kokkos::create_mirror(strain);
    Kokkos::deep_copy(strain_mirror, strain_host);
    Kokkos::deep_copy(strain, strain_mirror);

    const auto FC = Kokkos::View<double[6]>("FC");

    Kokkos::parallel_for(
        "CalculateForceFC", 1,
        KOKKOS_LAMBDA(size_t) { openturbine::beams::CalculateForceFC(Cuu, strain, FC); }
    );

    constexpr auto FC_exact_data = std::array{847., 2269., 3691., 5113., 6535., 7957.};
    const auto FC_exact =
        Kokkos::View<double[6], Kokkos::HostSpace>::const_type(FC_exact_data.data());

    const auto FC_mirror = Kokkos::create_mirror(FC);
    Kokkos::deep_copy(FC_mirror, FC);
    openturbine::tests::CompareWithExpected(FC_mirror, FC_exact);
}

}  // namespace

namespace openturbine::tests {

TEST(CalculateForceFCTests, OneNode) {
    TestCalculateForceFC();
}

}  // namespace openturbine::tests
