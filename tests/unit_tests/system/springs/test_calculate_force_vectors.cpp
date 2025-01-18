#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/springs/calculate_force_vectors.hpp"
#include "system/beams/test_calculate.hpp"

namespace {

void TestCalculateForceVectorsTests_ThreeElements() {
    const auto r = Kokkos::View<double[3][3]>("r");
    const auto c1 = Kokkos::View<double[3]>("c1");
    const auto f = Kokkos::View<double[3][3]>("f");

    constexpr auto r_data = std::array{
        1., 2., 3.,  // Element 1
        4., 5., 6.,  // Element 2
        7., 8., 9.   // Element 3
    };
    constexpr auto c1_data = std::array{2., -1., 0.5};  // Force coefficients

    const auto r_host = Kokkos::View<const double[3][3], Kokkos::HostSpace>(r_data.data());
    const auto c1_host = Kokkos::View<const double[3], Kokkos::HostSpace>(c1_data.data());

    const auto r_mirror = Kokkos::create_mirror(r);
    const auto c1_mirror = Kokkos::create_mirror(c1);

    Kokkos::deep_copy(r_mirror, r_host);
    Kokkos::deep_copy(c1_mirror, c1_host);

    Kokkos::deep_copy(r, r_mirror);
    Kokkos::deep_copy(c1, c1_mirror);

    Kokkos::parallel_for(
        "CalculateForceVectors", 3,
        KOKKOS_LAMBDA(const size_t i_elem) {
            openturbine::springs::CalculateForceVectors{i_elem, r, c1, f}();
        }
    );

    constexpr auto f_exact_data = std::array{
        2.,  4.,  6.,   // Element 1
        -4., -5., -6.,  // Element 2
        3.5, 4.,  4.5   // Element 3
    };
    const auto f_exact = Kokkos::View<const double[3][3], Kokkos::HostSpace>(f_exact_data.data());

    const auto f_result = Kokkos::create_mirror(f);
    Kokkos::deep_copy(f_result, f);

    openturbine::tests::CompareWithExpected(f_result, f_exact);
}
}  // namespace

namespace openturbine::tests {

TEST(CalculateForceVectorsTests, ThreeElements) {
    TestCalculateForceVectorsTests_ThreeElements();
}

}  // namespace openturbine::tests
