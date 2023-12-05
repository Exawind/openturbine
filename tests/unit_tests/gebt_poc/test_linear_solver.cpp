#include <limits>

#include <gtest/gtest.h>

#include "src/gebt_poc/linear_solver.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

TEST(GEBT_LinearSolverTest, Solve1x1Identity) {
    auto identity = openturbine::gen_alpha_solver::tests::create_diagonal_matrix({1});
    auto rhs = openturbine::gen_alpha_solver::create_vector({1});
    auto solution = Kokkos::View<double[1]>("solution");
    auto exact_solution = std::vector<double>{1.};

    openturbine::gebt_poc::solve_linear_system(identity, solution, rhs);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_EQ(solution_host(0), exact_solution[0]);
}

TEST(GEBT_LinearSolverTest, Solve3x3Identity) {
    auto identity = openturbine::gen_alpha_solver::tests::create_diagonal_matrix({1., 1., 1.});
    auto rhs = openturbine::gen_alpha_solver::create_vector({1., 2., 3.});
    auto solution = Kokkos::View<double[3]>("solution");
    auto exact_solution = std::vector<double>{1., 2., 3.};

    openturbine::gebt_poc::solve_linear_system(identity, solution, rhs);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_EQ(solution_host(0), exact_solution[0]);
    EXPECT_EQ(solution_host(1), exact_solution[1]);
    EXPECT_EQ(solution_host(2), exact_solution[2]);
}

TEST(GEBT_LinearSolverTest, Solve1x1Diagonal) {
    auto diagonal = openturbine::gen_alpha_solver::tests::create_diagonal_matrix({2.});
    auto rhs = openturbine::gen_alpha_solver::create_vector({1.});
    auto solution = Kokkos::View<double[1]>("solution");
    auto exact_solution = std::vector<double>{.5};

    openturbine::gebt_poc::solve_linear_system(diagonal, solution, rhs);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_NEAR(solution_host(0), exact_solution[0], std::numeric_limits<double>::epsilon());
}

TEST(GEBT_LinearSolverTest, Solve3x3Diagonal) {
    auto diagonal = openturbine::gen_alpha_solver::tests::create_diagonal_matrix({2., 8., 32.});
    auto rhs = openturbine::gen_alpha_solver::create_vector({1., 2., 4.});
    auto solution = Kokkos::View<double[3]>("solution");
    auto exact_solution = std::vector<double>{.5, .25, .125};

    openturbine::gebt_poc::solve_linear_system(diagonal, solution, rhs);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_NEAR(solution_host(0), exact_solution[0], std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(1), exact_solution[1], std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(2), exact_solution[2], std::numeric_limits<double>::epsilon());
}

TEST(GEBT_LinearSolverTest, Solve2x2Matrix) {
    auto matrix = openturbine::gen_alpha_solver::create_matrix({{1., 2.}, {3., 4.}});
    auto rhs = openturbine::gen_alpha_solver::create_vector({17., 39.});
    auto solution = Kokkos::View<double[2]>("solution");
    auto exact_solution = std::vector<double>{5., 6.};

    openturbine::gebt_poc::solve_linear_system(matrix, solution, rhs);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_NEAR(solution_host(0), exact_solution[0], 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(1), exact_solution[1], 10 * std::numeric_limits<double>::epsilon());
}

TEST(GEBT_LinearSolverTest, Solve3x3Matrix) {
    auto matrix =
        openturbine::gen_alpha_solver::create_matrix({{2., 6., 3.}, {4., -1., 3.}, {1., 3., 2.}});
    auto rhs = openturbine::gen_alpha_solver::create_vector({23., 11., 13.});
    auto solution = Kokkos::View<double[3]>("solution");
    auto exact_solution = std::vector<double>{1., 2., 3.};

    openturbine::gebt_poc::solve_linear_system(matrix, solution, rhs);

    auto solution_host = Kokkos::create_mirror(solution);
    Kokkos::deep_copy(solution_host, solution);
    EXPECT_NEAR(solution_host(0), exact_solution[0], 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(1), exact_solution[1], 10 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(solution_host(2), exact_solution[2], 10 * std::numeric_limits<double>::epsilon());
}

TEST(GEBT_LinearSolverTest, Solve2x2Matrix_fail) {
    auto matrix = openturbine::gen_alpha_solver::create_matrix({{0., 0.}, {0., 0.}});
    auto rhs = openturbine::gen_alpha_solver::create_vector({17., 39.});
    auto solution = Kokkos::View<double[2]>("solution");
    auto exact_solution = std::vector<double>{17., 39.};

    EXPECT_THROW(openturbine::gebt_poc::solve_linear_system(matrix, solution, rhs), std::runtime_error);
}