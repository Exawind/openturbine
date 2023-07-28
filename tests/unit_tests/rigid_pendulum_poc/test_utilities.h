#pragma once

#include "src/rigid_pendulum_poc/utilities.h"

#include "src/rigid_pendulum_poc/quaternion.h"

namespace openturbine::rigid_pendulum::tests {

HostView2D create_diagonal_matrix(const std::vector<double>& values);

// Check if members of the provided 1D Kokkos view is equal to the provided expected vector
void expect_kokkos_view_1D_equal(
    HostView1D, const std::vector<double>&, double epsilon = kTOLERANCE
);

// Check if members of the provided 2D Kokkos view is equal to the provided expected matrix
void expect_kokkos_view_2D_equal(
    HostView2D, const std::vector<std::vector<double>>&, double epsilon = kTOLERANCE
);

// Multiply a 3x3 rotation matrix with a provided 3x1 vector and return the result
Vector multiply_rotation_matrix_with_vector(const RotationMatrix&, const Vector&);

}  // namespace openturbine::rigid_pendulum::tests
