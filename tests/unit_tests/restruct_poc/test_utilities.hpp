#pragma once

#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

Kokkos::View<double**> create_diagonal_matrix(const std::vector<double>& values);

// Check if members of the provided 1D Kokkos view is equal to the provided expected vector
void expect_kokkos_view_1D_equal(
    Kokkos::View<double*>, const std::vector<double>&, double epsilon = kTestTolerance
);

// Check if members of the provided 2D Kokkos view is equal to the provided expected matrix
void expect_kokkos_view_2D_equal(
    Kokkos::View<double**>, const std::vector<std::vector<double>>&, double epsilon = kTestTolerance
);

// Check if members of the provided 3D Kokkos view is equal to the provided expected tensor
void expect_kokkos_view_3D_equal(
    Kokkos::View<double***> view, const std::vector<std::vector<std::vector<double>>>& expected,
    double epsilon = kTestTolerance
);

std::vector<double> kokkos_view_1D_to_vector(Kokkos::View<double*> view);
std::vector<std::vector<double>> kokkos_view_2D_to_vector(Kokkos::View<double**> view);

}  // namespace openturbine::restruct_poc::tests
