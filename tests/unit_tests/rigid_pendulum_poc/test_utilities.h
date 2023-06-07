#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum::tests {

using HostView1D = Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>;
using HostView2D = Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace>;

HostView2D create_diagonal_matrix(const std::vector<double>& values);

HostView1D create_vector(const std::vector<double>& values);

HostView2D create_matrix(const std::vector<std::vector<double>>& values);

// Check if members of the provided 1D Kokkos view is equal to the provided expected vector
void expect_kokkos_view_1D_equal(HostView1D, const std::vector<double>&);

}  // namespace openturbine::rigid_pendulum::tests
