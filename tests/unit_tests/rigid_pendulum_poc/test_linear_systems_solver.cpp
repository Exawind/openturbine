#include <limits>

#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/solver.h"

#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

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

    EXPECT_THROW(openturbine::rigid_pendulum::solve_linear_system(matrix_2x3, solution_2x1),
                 std::invalid_argument);
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

    EXPECT_THROW(openturbine::rigid_pendulum::solve_linear_system(matrix_5x3, solution_5x1),
                 std::invalid_argument);
}

TEST(LinearSolverTest, Check1x1Matrix2x1VectorCompatibility) {
    auto system_1x1 = create_diagonal_matrix({1.});
    auto solution_2x1 = create_vector({1., 2.});

    EXPECT_THROW(openturbine::rigid_pendulum::solve_linear_system(system_1x1, solution_2x1),
                 std::invalid_argument);
}

TEST(LinearSolverTest, Check3x3Matrix2x1VectorCompatibility) {
    auto system_3x3 = create_diagonal_matrix({1., 1., 1.});
    auto solution_2x1 = create_vector({1., 2.});

    EXPECT_THROW(openturbine::rigid_pendulum::solve_linear_system(system_3x3, solution_2x1),
                 std::invalid_argument);
}

}  // namespace openturbine::rigid_pendulum::tests
