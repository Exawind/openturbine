#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

namespace openturbine::tests {

Kokkos::View<double**> create_diagonal_matrix(const std::vector<double>& values);

// Check if members of the provided 1D Kokkos view is equal to the provided expected vector
void expect_kokkos_view_1D_equal(
    const Kokkos::View<const double*>&, const std::vector<double>&, double epsilon = 1.e-6
);

// Check if members of the provided 2D Kokkos view is equal to the provided expected matrix
void expect_kokkos_view_2D_equal(
    const Kokkos::View<const double**, Kokkos::LayoutStride>&,
    const std::vector<std::vector<double>>&, double epsilon = 1.e-6
);

// Check if members of the provided 3D Kokkos view is equal to the provided expected tensor
void expect_kokkos_view_3D_equal(
    const Kokkos::View<const double***, Kokkos::LayoutStride>& view,
    const std::vector<std::vector<std::vector<double>>>& expected, double epsilon = 1.e-6
);

template <typename T>
std::vector<T> kokkos_view_1D_to_vector(Kokkos::View<T*> view) {
    auto view_host = Kokkos::create_mirror(view);
    Kokkos::deep_copy(view_host, view);
    std::vector<T> values;
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        values.push_back(view_host(i));
    }
    return values;
}

std::vector<std::vector<double>> kokkos_view_2D_to_vector(const Kokkos::View<double**>& view);

}  // namespace openturbine::tests
