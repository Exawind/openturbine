#include <limits>

#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/solver.h"
#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(LinearSolverTest, Solve1x1Identity) {
    auto identity = create_diagonal_matrix({1});
    auto solution = create_vector({1});
    auto exact_solution = std::vector<double>{1.};

    openturbine::rigid_pendulum::solve_linear_system(identity, solution);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_EQ(solution_host(0), exact_solution[0]);
}

TEST(LinearSolverTest, Solve3x3Identity) {
    auto identity = create_diagonal_matrix({1., 1., 1.});
    auto solution = create_vector({1., 2., 3.});
    auto exact_solution = std::vector<double>{1., 2., 3.};

    openturbine::rigid_pendulum::solve_linear_system(identity, solution);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_EQ(solution_host(0), exact_solution[0]);
    EXPECT_EQ(solution_host(1), exact_solution[1]);
    EXPECT_EQ(solution_host(2), exact_solution[2]);
}

TEST(LinearSolverTest, Solve1x1Diagonal) {
    auto diagonal = create_diagonal_matrix({2.});
    auto solution = create_vector({1.});
    auto exact_solution = std::vector<double>{.5};

    openturbine::rigid_pendulum::solve_linear_system(diagonal, solution);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_NEAR(solution_host(0), exact_solution[0], std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, Solve3x3Diagonal) {
    auto diagonal = create_diagonal_matrix({2., 8., 32.});
    auto solution = create_vector({1., 2., 4.});
    auto exact_solution = std::vector<double>{.5, .25, .125};

    openturbine::rigid_pendulum::solve_linear_system(diagonal, solution);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_NEAR(solution_host(0), exact_solution[0], std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(1), exact_solution[1], std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(2), exact_solution[2], std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, Solve2x2Matrix) {
    auto matrix = create_matrix({{1., 2.}, {3., 4.}});
    auto solution = create_vector({17., 39.});
    auto exact_solution = std::vector<double>{5., 6.};

    openturbine::rigid_pendulum::solve_linear_system(matrix, solution);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_NEAR(solution_host(0), exact_solution[0], 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(1), exact_solution[1], 10 * std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, Solve3x3Matrix) {
    auto matrix = create_matrix({{2., 6., 3.}, {4., -1., 3.}, {1., 3., 2.}});
    auto solution = create_vector({23., 11., 13.});
    auto exact_solution = std::vector<double>{1., 2., 3.};

    openturbine::rigid_pendulum::solve_linear_system(matrix, solution);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_NEAR(solution_host(0), exact_solution[0], 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(1), exact_solution[1], 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(2), exact_solution[2], 10 * std::numeric_limits<double>::epsilon());
}

}  // namespace openturbine::rigid_pendulum::tests
