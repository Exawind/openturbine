#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/test_calculate.hpp"
#include "system/springs/calculate_force_vectors.hpp"

namespace {

void TestCalculateForceVectorsTests_ThreeElements() {
    const auto f0 = Kokkos::View<double[3]>("f0");
    const auto f1 = Kokkos::View<double[3]>("f1");
    const auto f2 = Kokkos::View<double[3]>("f2");

    Kokkos::parallel_for(
        "CalculateForceVectors", 1,
        KOKKOS_LAMBDA(const size_t) {
            constexpr auto r0_data = Kokkos::Array<double, 3>{1., 2., 3.};
            constexpr auto r1_data = Kokkos::Array<double, 3>{4., 5., 6.};
            constexpr auto r2_data = Kokkos::Array<double, 3>{7., 8., 9.};

            const auto r0 = Kokkos::View<double[3]>::const_type(r0_data.data());
            const auto r1 = Kokkos::View<double[3]>::const_type(r1_data.data());
            const auto r2 = Kokkos::View<double[3]>::const_type(r2_data.data());

            constexpr auto c10 = 2.;
            constexpr auto c11 = -1.;
            constexpr auto c12 = 0.5;

            openturbine::springs::CalculateForceVectors(r0, c10, f0);
            openturbine::springs::CalculateForceVectors(r1, c11, f1);
            openturbine::springs::CalculateForceVectors(r2, c12, f2);
        }
    );

    constexpr auto f0_exact_data = std::array{2., 4., 6.};
    constexpr auto f1_exact_data = std::array{-4., -5., -6.};
    constexpr auto f2_exact_data = std::array{3.5, 4., 4.5};

    const auto f0_exact =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(f0_exact_data.data());
    const auto f1_exact =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(f1_exact_data.data());
    const auto f2_exact =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(f2_exact_data.data());

    const auto f0_result = Kokkos::create_mirror(f0);
    const auto f1_result = Kokkos::create_mirror(f1);
    const auto f2_result = Kokkos::create_mirror(f2);

    Kokkos::deep_copy(f0_result, f0);
    Kokkos::deep_copy(f1_result, f1);
    Kokkos::deep_copy(f2_result, f2);

    openturbine::tests::CompareWithExpected(f0_result, f0_exact);
    openturbine::tests::CompareWithExpected(f1_result, f1_exact);
    openturbine::tests::CompareWithExpected(f2_result, f2_exact);
}
}  // namespace

namespace openturbine::tests {

TEST(CalculateForceVectorsTests, ThreeElements) {
    TestCalculateForceVectorsTests_ThreeElements();
}

}  // namespace openturbine::tests
