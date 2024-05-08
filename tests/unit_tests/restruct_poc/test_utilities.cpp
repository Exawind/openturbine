#include "test_utilities.hpp"

#include <limits>

#include <gtest/gtest.h>

namespace openturbine::gen_alpha_solver::tests {

Kokkos::View<double**> create_diagonal_matrix(const std::vector<double>& values) {
    auto matrix = Kokkos::View<double**>("matrix", values.size(), values.size());
    auto matrix_host = Kokkos::create_mirror(matrix);

    for (size_t index = 0; index < values.size(); ++index) {
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

    Kokkos::View<double**> view_contiguous("view_contiguous", view.extent(0), view.extent(1));
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror(view_contiguous);
    Kokkos::deep_copy(view_host, view_contiguous);
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        for (size_t j = 0; j < view_host.extent(1); ++j) {
            EXPECT_NEAR(view_host(i, j), expected[i][j], epsilon);
        }
    }
}

std::vector<double> kokkos_view_1D_to_vector(Kokkos::View<double*> view) {
    auto view_host = Kokkos::create_mirror(view);
    Kokkos::deep_copy(view_host, view);
    std::vector<double> values;
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        values.push_back(view_host(i));
    }
    return values;
}

std::vector<std::vector<double>> kokkos_view_2D_to_vector(Kokkos::View<double**> view) {
    Kokkos::View<double**> view_contiguous("view_contiguous", view.extent(0), view.extent(1));
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror(view_contiguous);
    Kokkos::deep_copy(view_host, view_contiguous);
    std::vector<std::vector<double>> values(view.extent(0));
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        for (size_t j = 0; j < view_host.extent(1); ++j) {
            values[i].push_back(view_host(i, j));
        }
    }
    return values;
}

void expect_kokkos_view_3D_equal(
    Kokkos::View<double***> view, const std::vector<std::vector<std::vector<double>>>& expected,
    double epsilon
) {
    EXPECT_EQ(view.extent(0), expected.size());
    EXPECT_EQ(view.extent(1), expected.front().size());
    EXPECT_EQ(view.extent(2), expected.front().front().size());

    Kokkos::View<double***> view_contiguous(
        "view_contiguous", view.extent(0), view.extent(1), view.extent(2)
    );
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror(view_contiguous);
    Kokkos::deep_copy(view_host, view_contiguous);
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        for (size_t j = 0; j < view_host.extent(1); ++j) {
            for (size_t k = 0; k < view_host.extent(2); ++k) {
                EXPECT_NEAR(view_host(i, j, k), expected[i][j][k], epsilon);
            }
        }
    }
}

Vector multiply_rotation_matrix_with_vector(const RotationMatrix& R, const Vector& v) {
    return Vector{
        R(0, 0) * v.GetXComponent() + R(0, 1) * v.GetYComponent() + R(0, 2) * v.GetZComponent(),
        R(1, 0) * v.GetXComponent() + R(1, 1) * v.GetYComponent() + R(1, 2) * v.GetZComponent(),
        R(2, 0) * v.GetXComponent() + R(2, 1) * v.GetYComponent() + R(2, 2) * v.GetZComponent(),
    };
}

}  // namespace openturbine::gen_alpha_solver::tests
