#include "gtest/gtest.h"

#include <Kokkos_Core.hpp>

#include "src/rigid_pendulum_poc/solver.h"

namespace openturbine::rigid_pendulum::tests {

Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace> 
create_diagonal_matrix(std::vector<double> diagonal)
{
  auto matrix = Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace>("identity", diagonal.size(), diagonal.size());
  Kokkos::deep_copy(matrix, 0);

  for(auto& entry : diagonal)
  {
    auto index = &entry - &diagonal[0];
    matrix(index, index) = entry;
  }

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

}  // namespace oturb_tests

