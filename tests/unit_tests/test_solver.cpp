#include <limits>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "src/rigid_pendulum_poc/solver.h"

namespace openturbine::rigid_pendulum::tests {

Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace> 
create_diagonal_matrix(std::vector<double> diagonal)
{
  auto matrix = Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace>("identity", diagonal.size(), diagonal.size());
  auto diagonal_entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, diagonal.size());
  auto fill_diagonal = [matrix, diagonal](int index) { matrix(index, index) = diagonal[index]; };

  Kokkos::deep_copy(matrix, 0);
  Kokkos::parallel_for(diagonal_entries, fill_diagonal);

  return matrix;
}

TEST(LinearSolverTest, solve_1x1_identity)
{
  auto identity = create_diagonal_matrix({1});
  auto solution = Kokkos::View<double*>("solution", 1);
  auto exactSolution = Kokkos::View<double*>("exact solution", 1);

  solution(0) = 1.;
  
  Kokkos::deep_copy(exactSolution, solution);

  openturbine::rigid_pendulum::solve_linear_system(identity, solution);

  EXPECT_EQ(solution(0), exactSolution(0));
}

TEST(LinearSolverTest, solve_3x3_identity)
{
  auto identity = create_diagonal_matrix({1., 1., 1.});
  auto solution = Kokkos::View<double*>("solution", 3);
  auto exactSolution = Kokkos::View<double*>("exact solution", 3);

  solution(0) = 1.;
  solution(1) = 2.;
  solution(2) = 3.;
  
  Kokkos::deep_copy(exactSolution, solution);

  openturbine::rigid_pendulum::solve_linear_system(identity, solution);

  EXPECT_EQ(solution(0), exactSolution(0));
  EXPECT_EQ(solution(1), exactSolution(1));
  EXPECT_EQ(solution(2), exactSolution(2));
}

TEST(LinearSolverTest, solve_1x1_diagonal)
{
  auto identity = create_diagonal_matrix({2.});
  auto solution = Kokkos::View<double*>("solution", 3);
  auto exactSolution = Kokkos::View<double*>("exact solution", 3);

  solution(0) = 1.;
  
  exactSolution(0) = .5;

  openturbine::rigid_pendulum::solve_linear_system(identity, solution);

  EXPECT_NEAR(solution(0), exactSolution(0), std::numeric_limits<double>::epsilon());
}

TEST(LinearSolverTest, solve_3x3_diagonal)
{
  auto identity = create_diagonal_matrix({2., 8., 32.});
  auto solution = Kokkos::View<double*>("solution", 3);
  auto exactSolution = Kokkos::View<double*>("exact solution", 3);

  solution(0) = 1.;
  solution(1) = 2.;
  solution(2) = 4.;
  
  exactSolution(0) = .5;
  exactSolution(1) = .25;
  exactSolution(2) = .125;

  openturbine::rigid_pendulum::solve_linear_system(identity, solution);

  EXPECT_NEAR(solution(0), exactSolution(0), std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(solution(1), exactSolution(1), std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(solution(2), exactSolution(2), std::numeric_limits<double>::epsilon());
}

}  // namespace oturb_tests

