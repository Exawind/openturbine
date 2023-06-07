#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

#include <gtest/gtest.h>

#include <limits>

namespace openturbine::rigid_pendulum::tests {

HostView2D create_diagonal_matrix(const std::vector<double>& values) {
    auto matrix = HostView2D("matrix", values.size(), values.size());
    auto diagonal_entries =
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, values.size());
    auto fill_diagonal = [matrix, values](int index) {
        matrix(index, index) = values[index];
    };

    Kokkos::deep_copy(matrix, 0);
    Kokkos::parallel_for(diagonal_entries, fill_diagonal);

    return matrix;
}

HostView1D create_vector(const std::vector<double>& values) {
    auto vector = HostView1D("vector", values.size());
    auto entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, values.size());
    auto fill_vector = [vector, values](int index) {
        vector(index) = values[index];
    };

    Kokkos::parallel_for(entries, fill_vector);

    return vector;
}

HostView2D create_matrix(const std::vector<std::vector<double>>& values) {
    auto matrix = HostView2D("matrix", values.size(), values.front().size());
    auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
        {0, 0}, {values.size(), values.front().size()});
    auto fill_matrix = [matrix, values](int row, int column) {
        matrix(row, column) = values[row][column];
    };

    Kokkos::parallel_for(entries, fill_matrix);

    return matrix;
}

void expect_kokkos_view_1D_equal(HostView1D view, const std::vector<double>& expected) {
    for (size_t i = 0; i < view.extent(0); ++i) {
        ASSERT_NEAR(view(i), expected[i], std::numeric_limits<double>::epsilon());
    }
}

}  // namespace openturbine::rigid_pendulum::tests
