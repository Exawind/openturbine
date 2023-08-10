#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

#include <limits>

#include <gtest/gtest.h>

namespace openturbine::gen_alpha_solver::tests {

HostView2D create_diagonal_matrix(const std::vector<double>& values) {
    auto matrix = HostView2D("matrix", values.size(), values.size());
    auto diagonal_entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, values.size());
    auto fill_diagonal = [matrix, values](int index) {
        matrix(index, index) = values[index];
    };

    Kokkos::parallel_for(diagonal_entries, fill_diagonal);

    return matrix;
}

void expect_kokkos_view_1D_equal(
    HostView1D view, const std::vector<double>& expected, double epsilon
) {
    EXPECT_EQ(view.extent(0), expected.size());
    for (size_t i = 0; i < view.extent(0); ++i) {
        EXPECT_NEAR(view(i), expected[i], epsilon);
    }
}

void expect_kokkos_view_2D_equal(
    HostView2D view, const std::vector<std::vector<double>>& expected, double epsilon
) {
    EXPECT_EQ(view.extent(0), expected.size());
    EXPECT_EQ(view.extent(1), expected.front().size());
    for (size_t i = 0; i < view.extent(0); ++i) {
        for (size_t j = 0; j < view.extent(1); ++j) {
            EXPECT_NEAR(view(i, j), expected[i][j], epsilon);
        }
    }
}

Vector multiply_rotation_matrix_with_vector(const RotationMatrix& R, const Vector& v) {
    return Vector{
        std::get<0>(R).GetXComponent() * v.GetXComponent() +
            std::get<0>(R).GetYComponent() * v.GetYComponent() +
            std::get<0>(R).GetZComponent() * v.GetZComponent(),
        std::get<1>(R).GetXComponent() * v.GetXComponent() +
            std::get<1>(R).GetYComponent() * v.GetYComponent() +
            std::get<1>(R).GetZComponent() * v.GetZComponent(),
        std::get<2>(R).GetXComponent() * v.GetXComponent() +
            std::get<2>(R).GetYComponent() * v.GetYComponent() +
            std::get<2>(R).GetZComponent() * v.GetZComponent(),
    };
};

}  // namespace openturbine::gen_alpha_solver::tests
