#pragma once

#include <Kokkos_Core.hpp>

#include "src/rigid_pendulum_poc/quaternion.h"

namespace openturbine::rigid_pendulum::tests {

using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;
using HostView2D = Kokkos::View<double**, Kokkos::HostSpace>;

HostView2D create_diagonal_matrix(const std::vector<double>& values);

HostView1D create_vector(const std::vector<double>& values);

HostView2D create_matrix(const std::vector<std::vector<double>>& values);

// Check if members of the provided 1D Kokkos view is equal to the provided expected vector
void expect_kokkos_view_1D_equal(HostView1D, const std::vector<double>&);

// Multiply a 3x3 rotation matrix with a provided 3x1 vector and return the result
Vector multiply_rotation_matrix_with_vector(const RotationMatrix&, const Vector&);

}  // namespace openturbine::rigid_pendulum::tests
