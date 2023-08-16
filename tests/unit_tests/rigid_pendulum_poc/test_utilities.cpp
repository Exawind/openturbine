#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

#include <limits>

#include <gtest/gtest.h>

namespace openturbine::rigid_pendulum::tests {

Kokkos::View<double**> create_diagonal_matrix(const std::vector<double>& values) {
    auto matrix = Kokkos::View<double**>("matrix", values.size(), values.size());
    auto matrix_host = Kokkos::create_mirror(matrix);

    for(size_t index = 0; index < values.size(); ++index) {
        matrix_host(index, index) = values[index];
    }
    Kokkos::deep_copy(matrix, matrix_host);

    return matrix;
}

void expect_kokkos_view_1D_equal(
    Kokkos::View<double*> view, const std::vector<double>& expected, double epsilon
) {
    EXPECT_EQ(view.extent(0), expected.size());
    auto view_host = Kokkos::create_mirror(view);
    Kokkos::deep_copy(view_host, view);
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        EXPECT_NEAR(view_host(i), expected[i], epsilon);
    }
}

void expect_kokkos_view_2D_equal(
    Kokkos::View<double**> view, const std::vector<std::vector<double>>& expected, double epsilon
) {
    EXPECT_EQ(view.extent(0), expected.size());
    EXPECT_EQ(view.extent(1), expected.front().size());
    auto view_host = Kokkos::create_mirror(view);
    Kokkos::deep_copy(view_host, view);
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        for (size_t j = 0; j < view_host.extent(1); ++j) {
            EXPECT_NEAR(view_host(i, j), expected[i][j], epsilon);
        }
    }
}

Vector multiply_rotation_matrix_with_vector(const RotationMatrix& R, const Vector& v) {
    return Vector{
        R(0,0) * v.GetXComponent() +
            R(0,1) * v.GetYComponent() +
            R(0,2) * v.GetZComponent(),
        R(1,0) * v.GetXComponent() +
            R(1,1) * v.GetYComponent() +
            R(1,2) * v.GetZComponent(),
        R(2,0) * v.GetXComponent() +
            R(2,1) * v.GetYComponent() +
            R(2,2) * v.GetZComponent(),
    };
};

}  // namespace openturbine::rigid_pendulum::tests
