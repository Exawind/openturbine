#include "src/rigid_pendulum_poc/solver.h"

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_abs.hpp> // For absolute value operations
#include <KokkosBlas1_axpby.hpp> // For linear combination operations
#include <KokkosBlas1_dot.hpp> // For dot product operations
#include <KokkosBlas1_nrm2.hpp> // For norm operations

#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

LinearSolver::LinearSolver(SolverType solver_type)
    : solver_type_(solver_type) {
}

Kokkos::View<double*> LinearSolver::Solve(const Kokkos::View<double*>& stiffness_matrix,
    const Kokkos::View<double*>& load_vector) const {
    return load_vector;
}

}  // namespace openturbine::rigid_pendulum

