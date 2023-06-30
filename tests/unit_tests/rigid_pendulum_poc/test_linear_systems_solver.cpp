#include <limits>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/solver.h"

namespace openturbine::rigid_pendulum::tests {

Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace> create_diagonal_matrix(
    const std::vector<double>& values
) {
    auto matrix = Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace>(
        "matrix", values.size(), values.size()
    );
    auto diagonal_entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, values.size());
    auto fill_diagonal = [matrix, values](int index) {
        matrix(index, index) = values[index];
    };

    Kokkos::deep_copy(matrix, 0);
    Kokkos::parallel_for(diagonal_entries, fill_diagonal);

    return matrix;
}

Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> create_vector(
    const std::vector<double>& values
) {
    auto vector = Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>("vector", values.size());
    auto entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, values.size());
    auto fill_vector = [vector, values](int index) {
        vector(index) = values[index];
    };

    Kokkos::parallel_for(entries, fill_vector);

    return vector;
}

Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace> create_matrix(
    const std::vector<std::vector<double>>& values
) {
    auto matrix = Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace>(
        "matrix", values.size(), values.front().size()
    );
    auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
        {0, 0}, {values.size(), values.front().size()}
    );
    auto fill_matrix = [matrix, values](int row, int column) {
        matrix(row, column) = values[row][column];
    };

    Kokkos::parallel_for(entries, fill_matrix);

    return matrix;
}

TEST(LinearSolverTest, Solve1x1Identity) {
    auto identity = create_diagonal_matrix({1});
    auto solution = create_vector({1});
    auto exact_solution = create_vector({1});

    openturbine::rigid_pendulum::solve_linear_system(identity, solution);

    EXPECT_EQ(solution(0), exact_solution(0));
}

TEST(LinearSolverTest, Solve3x3Identity) {
    auto identity = create_diagonal_matrix({1., 1., 1.});
    auto solution = create_vector({1., 2., 3.});
    auto exact_solution = create_vector({1., 2., 3.});

    Kokkos::deep_copy(exact_solution, solution);

    openturbine::rigid_pendulum::solve_linear_system(identity, solution);

    EXPECT_EQ(solution(0), exact_solution(0));
    EXPECT_EQ(solution(1), exact_solution(1));
    EXPECT_EQ(solution(2), exact_solution(2));
}

TEST(LinearSolverTest, Solve1x1Diagonal) {
    auto diagonal = create_diagonal_matrix({2.});
    auto solution = create_vector({1.});
    auto exact_solution = create_vector({.5});

    openturbine::rigid_pendulum::solve_linear_system(diagonal, solution);

    EXPECT_NEAR(solution(0), exact_solution(0), std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, Solve3x3Diagonal) {
    auto diagonal = create_diagonal_matrix({2., 8., 32.});
    auto solution = create_vector({1., 2., 4.});
    auto exact_solution = create_vector({.5, .25, .125});

    openturbine::rigid_pendulum::solve_linear_system(diagonal, solution);

    EXPECT_NEAR(solution(0), exact_solution(0), std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution(1), exact_solution(1), std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution(2), exact_solution(2), std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, Solve2x2Matrix) {
    auto matrix = create_matrix({{1., 2.}, {3., 4.}});
    auto solution = create_vector({17., 39.});
    auto exact_solution = create_vector({5., 6.});

    openturbine::rigid_pendulum::solve_linear_system(matrix, solution);

    EXPECT_NEAR(solution(0), exact_solution(0), 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution(1), exact_solution(1), 10 * std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, Solve3x3Matrix) {
    auto matrix = create_matrix({{2., 6., 3.}, {4., -1., 3.}, {1., 3., 2.}});
    auto solution = create_vector({23., 11., 13.});
    auto exact_solution = create_vector({1., 2., 3.});

    openturbine::rigid_pendulum::solve_linear_system(matrix, solution);

    EXPECT_NEAR(solution(0), exact_solution(0), 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution(1), exact_solution(1), 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution(2), exact_solution(2), 10 * std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, Check2x3MatrixShape) {
    auto matrix_2x3 = create_matrix({{1., 2., 3.}, {4., 5., 6.}});
    auto solution_2x1 = create_vector({1., 1.});

    EXPECT_THROW(
        openturbine::rigid_pendulum::solve_linear_system(matrix_2x3, solution_2x1),
        std::invalid_argument
    );
}

TEST(LinearSolverTest, Check5x3MatrixShape) {
    auto matrix_5x3 = create_matrix({
        {1., 2., 3.},
        {4., 5., 6.},
        {7., 8., 9.},
        {10., 11., 12.},
        {13., 14., 15.},
    });
    auto solution_5x1 = create_vector({1., 1., 1., 1., 1.});

    EXPECT_THROW(
        openturbine::rigid_pendulum::solve_linear_system(matrix_5x3, solution_5x1),
        std::invalid_argument
    );
}

TEST(LinearSolverTest, Check1x1Matrix2x1VectorCompatibility) {
    auto system_1x1 = create_diagonal_matrix({1.});
    auto solution_2x1 = create_vector({1., 2.});

    EXPECT_THROW(
        openturbine::rigid_pendulum::solve_linear_system(system_1x1, solution_2x1),
        std::invalid_argument
    );
}

TEST(LinearSolverTest, Check3x3Matrix2x1VectorCompatibility) {
    auto system_3x3 = create_diagonal_matrix({1., 1., 1.});
    auto solution_2x1 = create_vector({1., 2.});

    EXPECT_THROW(
        openturbine::rigid_pendulum::solve_linear_system(system_3x3, solution_2x1),
        std::invalid_argument
    );
}

}  // namespace openturbine::rigid_pendulum::tests
