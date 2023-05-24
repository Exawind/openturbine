#include "gtest/gtest.h"

#include <Kokkos_Core.hpp>

#include "src/rigid_pendulum_poc/solver.h"

namespace openturbine::rigid_pendulum::tests {

TEST(LinearSolverTest, solve_1x1_identity)
{
  auto identity = Kokkos::View<double**>("identity", 1, 1);
  auto solution = Kokkos::View<double*>("solution", 1);
  auto exactSolution = Kokkos::View<double*>("exact solution", 1);

  identity(0, 0) = 1.;
  solution(0) = 1.;
  
  Kokkos::deep_copy(exactSolution, solution);

  openturbine::rigid_pendulum::solve_linear_system(identity, solution);

  EXPECT_EQ(solution(0), exactSolution(0));
}

TEST(LinearSolverTest, solve_3x3_identity)
{
  auto identity = Kokkos::View<double**>("identity", 3, 3);
  auto solution = Kokkos::View<double*>("solution", 3);
  auto exactSolution = Kokkos::View<double*>("exact solution", 3);

  Kokkos::deep_copy(identity, 0);
  identity(0, 0) = 1.;
  identity(1, 1) = 1.;
  identity(2, 2) = 1.;

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

