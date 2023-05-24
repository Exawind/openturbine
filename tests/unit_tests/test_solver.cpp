#include <limits>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "src/rigid_pendulum_poc/solver.h"

namespace openturbine::rigid_pendulum::tests {

Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace> 
create_diagonal_matrix(std::vector<double> values)
{
  auto matrix = Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace>("matrix", values.size(), values.size());
  auto diagonal_entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, values.size());
  auto fill_diagonal = [matrix, values](int index) { matrix(index, index) = values[index]; };

  Kokkos::deep_copy(matrix, 0);
  Kokkos::parallel_for(diagonal_entries, fill_diagonal);

  return matrix;
}

Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
create_vector(std::vector<double> values)
{
  auto vector = Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>("vector", values.size());
  auto entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, values.size());
  auto fill_vector = [vector, values](int index) { vector(index) = values[index]; };

  Kokkos::parallel_for(entries, fill_vector);

  return vector;
}

Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace> 
create_matrix(std::vector<std::vector<double>> values)
{
  auto matrix = Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace>("matrix", values.size(), values.size());
  auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>({0, 0}, {values.size(), values.front().size()});
  auto fill_matrix = [matrix, values](int row, int column) { matrix(row, column) = values[row][column]; };

  Kokkos::parallel_for(entries, fill_matrix);

  return matrix;
}

TEST(LinearSolverTest, solve_1x1_identity)
{
  auto identity = create_diagonal_matrix({1});
  auto solution = create_vector({1});
  auto exactSolution = create_vector({1});

  auto info = openturbine::rigid_pendulum::solve_linear_system(identity, solution);

  ASSERT_EQ(info, 0);

  EXPECT_EQ(solution(0), exactSolution(0));
}

TEST(LinearSolverTest, solve_3x3_identity)
{
  auto identity = create_diagonal_matrix({1., 1., 1.});
  auto solution = create_vector({1., 2., 3.});
  auto exactSolution = create_vector({1., 2., 3.});
  
  Kokkos::deep_copy(exactSolution, solution);

  auto info = openturbine::rigid_pendulum::solve_linear_system(identity, solution);

  ASSERT_EQ(info, 0);

  EXPECT_EQ(solution(0), exactSolution(0));
  EXPECT_EQ(solution(1), exactSolution(1));
  EXPECT_EQ(solution(2), exactSolution(2));
}

TEST(LinearSolverTest, solve_1x1_diagonal)
{
  auto diagonal = create_diagonal_matrix({2.});
  auto solution = create_vector({1.});
  auto exactSolution = create_vector({.5});

  auto info = openturbine::rigid_pendulum::solve_linear_system(diagonal, solution);

  ASSERT_EQ(info, 0);

  EXPECT_NEAR(solution(0), exactSolution(0), std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, solve_3x3_diagonal)
{
  auto diagonal = create_diagonal_matrix({2., 8., 32.});
  auto solution = create_vector({1., 2., 4.});
  auto exactSolution = create_vector({.5, .25, .125});

  auto info = openturbine::rigid_pendulum::solve_linear_system(diagonal, solution);

  ASSERT_EQ(info, 0);

  EXPECT_NEAR(solution(0), exactSolution(0), std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(solution(1), exactSolution(1), std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(solution(2), exactSolution(2), std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, solve_2x2_matrix)
{
  auto matrix = create_matrix({{1., 2.}, {3., 4.}});
  auto solution = create_vector({17., 39.});
  auto exactSolution = create_vector({5., 6.});

  auto info = openturbine::rigid_pendulum::solve_linear_system(matrix, solution);

  ASSERT_EQ(info, 0);

  EXPECT_NEAR(solution(0), exactSolution(0), 10*std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(solution(1), exactSolution(1), 10*std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, solve_3x3_matrix)
{
  auto matrix = create_matrix({{2., 6., 3.}, {4., -1., 3.}, {1., 3., 2.}});
  auto solution = create_vector({23., 11., 13.});
  auto exactSolution = create_vector({1., 2., 3.});

  auto info = openturbine::rigid_pendulum::solve_linear_system(matrix, solution);

  ASSERT_EQ(info, 0);

  EXPECT_NEAR(solution(0), exactSolution(0), 10*std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(solution(1), exactSolution(1), 10*std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(solution(2), exactSolution(2), 10*std::numeric_limits<double>::epsilon());
}

}  // namespace oturb_tests

