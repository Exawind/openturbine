#pragma once

#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gen_alpha_solver::tests {

Kokkos::View<double**> create_diagonal_matrix(const std::vector<double>& values);

// Check if members of the provided 1D Kokkos view is equal to the provided expected vector
void expect_kokkos_view_1D_equal(
    Kokkos::View<double*>, const std::vector<double>&, double epsilon = kTolerance
);

// Check if members of the provided 2D Kokkos view is equal to the provided expected matrix
void expect_kokkos_view_2D_equal(
    Kokkos::View<double**>, const std::vector<std::vector<double>>&, double epsilon = kTolerance
);

// Check if members of the provided 3D Kokkos view is equal to the provided expected tensor
void expect_kokkos_view_3D_equal(
    Kokkos::View<double***> view, const std::vector<std::vector<std::vector<double>>>& expected,
    double epsilon = kTolerance
);

// void expect_kokkos_view_2D_equal(
//     Kokkos::View<const double**>, const std::vector<std::vector<double>>&,
//     double epsilon = kTolerance
// );

// Multiply a 3x3 rotation matrix with a provided 3x1 vector and return the result
Vector multiply_rotation_matrix_with_vector(const RotationMatrix&, const Vector&);

std::vector<double> kokkos_view_1D_to_vector(Kokkos::View<double*> view);
std::vector<std::vector<double>> kokkos_view_2D_to_vector(Kokkos::View<double**> view);

}  // namespace openturbine::gen_alpha_solver::tests
