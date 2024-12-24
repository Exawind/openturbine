#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/system/springs/calculate_stiffness_matrix.hpp"
#include "tests/unit_tests/system/beams/test_calculate.hpp"

namespace openturbine::tests {

TEST(CalculateStiffnessMatrixTests, OneElement) {
    const auto c1 = Kokkos::View<double[1]>("c1");
    const auto c2 = Kokkos::View<double[1]>("c2");
    const auto r = Kokkos::View<double[1][3]>("r");
    const auto l = Kokkos::View<double[1]>("l");
    auto r_tilde = Kokkos::View<double[1][3][3]>("r_tilde");
    auto a = Kokkos::View<double[1][3][3]>("a");

    constexpr auto c1_data = std::array{2.};         // force calc. coeff 1
    constexpr auto c2_data = std::array{1.};         // force calc. coeff 2
    constexpr auto r_data = std::array{1., 2., 3.};  // distance vector
    constexpr auto l_data = std::array{1.};          // current length

    const auto c1_host = Kokkos::View<const double[1], Kokkos::HostSpace>(c1_data.data());
    const auto c2_host = Kokkos::View<const double[1], Kokkos::HostSpace>(c2_data.data());
    const auto r_host = Kokkos::View<const double[1][3], Kokkos::HostSpace>(r_data.data());
    const auto l_host = Kokkos::View<const double[1], Kokkos::HostSpace>(l_data.data());

    const auto c1_mirror = Kokkos::create_mirror(c1);
    const auto c2_mirror = Kokkos::create_mirror(c2);
    const auto r_mirror = Kokkos::create_mirror(r);
    const auto l_mirror = Kokkos::create_mirror(l);

    Kokkos::deep_copy(c1_mirror, c1_host);
    Kokkos::deep_copy(c2_mirror, c2_host);
    Kokkos::deep_copy(r_mirror, r_host);
    Kokkos::deep_copy(l_mirror, l_host);

    Kokkos::deep_copy(c1, c1_mirror);
    Kokkos::deep_copy(c2, c2_mirror);
    Kokkos::deep_copy(r, r_mirror);
    Kokkos::deep_copy(l, l_mirror);

    Kokkos::parallel_for(
        "CalculateStiffnessMatrix", 1, CalculateStiffnessMatrix{c1, c2, r, l, r_tilde, a}
    );

    // Expected r_tilde matrix (skew-symmetric matrix from r vector)
    constexpr auto r_tilde_exact_data = std::array{
        0.,  -3., 2.,   // row 1
        3.,  0.,  -1.,  // row 2
        -2., 1.,  0.    // row 3
    };

    // Expected a matrix: diag_term * I - c2 * r_tilde * r_tilde
    // where diag_term = c1 - c2 * l^2 = 2. - 1. * 1. = 1.
    constexpr auto a_exact_data = std::array{
        14., -2., -3.,  // row 1
        -2., 11., -6.,  // row 2
        -3., -6., 6.    // row 3
    };

    const auto r_tilde_exact =
        Kokkos::View<const double[1][3][3], Kokkos::HostSpace>(r_tilde_exact_data.data());
    const auto a_exact = Kokkos::View<const double[1][3][3], Kokkos::HostSpace>(a_exact_data.data());

    const auto r_tilde_result = Kokkos::create_mirror(r_tilde);
    const auto a_result = Kokkos::create_mirror(a);
    Kokkos::deep_copy(r_tilde_result, r_tilde);
    Kokkos::deep_copy(a_result, a);

    CompareWithExpected(r_tilde_result, r_tilde_exact);
    CompareWithExpected(a_result, a_exact);
}

}  // namespace openturbine::tests
