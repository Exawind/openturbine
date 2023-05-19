#include "gtest/gtest.h"

#include <Kokkos_Core.hpp>

#include "src/rigid_pendulum_poc/solver.h"

namespace oturb_tests {

using namespace openturbine::rigid_pendulum;

TEST(SolverType, DefaultValue) {
    // Test that the default value of a SolverType object is kNone
    SolverType type {};
    EXPECT_EQ(type, SolverType::kNone);
}

TEST(SolverType, Comparison) {
    // Test that different solver types can be compared
    EXPECT_LT(SolverType::kNone, SolverType::kDirectLinearSolver);
    EXPECT_EQ(SolverType::kDirectLinearSolver, SolverType::kDirectLinearSolver);
    EXPECT_NE(SolverType::kDirectLinearSolver, SolverType::kNone);
    EXPECT_LT(SolverType::kNone, SolverType::kIterativeLinearSolver);
    EXPECT_EQ(SolverType::kIterativeLinearSolver, SolverType::kIterativeLinearSolver);
    EXPECT_NE(SolverType::kIterativeLinearSolver, SolverType::kDirectLinearSolver);
    EXPECT_NE(SolverType::kIterativeLinearSolver, SolverType::kNone);
    EXPECT_LT(SolverType::kNone, SolverType::kIterativeNonlinearSolver);
    EXPECT_EQ(SolverType::kIterativeNonlinearSolver, SolverType::kIterativeNonlinearSolver);
    EXPECT_NE(SolverType::kIterativeNonlinearSolver, SolverType::kIterativeLinearSolver);
    EXPECT_NE(SolverType::kIterativeNonlinearSolver, SolverType::kDirectLinearSolver);
    EXPECT_NE(SolverType::kIterativeNonlinearSolver, SolverType::kNone);
}

class LinearSolverTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        Kokkos::initialize();
    }

    virtual void TearDown() {
        Kokkos::finalize();
    }
};

TEST_F(LinearSolverTest, Test1) {
    LinearSolver solver {};
    EXPECT_EQ(solver.GetSolverType(), SolverType::kNone);

    Kokkos::View<double*> x("x", 1);
    x[0] = 1.0;
    Kokkos::View<double*> A("A", 1);
    A[0] = 1.0;
    Kokkos::View<double*> b("b", 1);
    Kokkos::deep_copy(b, solver.Solve(A, x));

    ASSERT_EQ(b(0), 1.);
}

}  // namespace oturb_tests

