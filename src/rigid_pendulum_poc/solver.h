#pragma once

#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

/// @brief Enumerate different types of solvers
enum class SolverType {
    kNone = 0,
    kDirectLinearSolver = 1,
    kIterativeLinearSolver = 2,
    kIterativeNonlinearSolver = 3
};

/// @brief A class for solving linear systems
class LinearSolver {
public:
    LinearSolver(SolverType=SolverType::kNone);

    SolverType GetSolverType() const { return solver_type_; }

    Kokkos::View<double*> Solve(const Kokkos::View<double*>&, const Kokkos::View<double*>&) const;

private:
    SolverType solver_type_;
};


}  // namespace openturbine::rigid_pendulum

