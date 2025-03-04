#pragma once

#include <Kokkos_Core.hpp>

#include "solver/solver.hpp"

namespace openturbine {

inline void ResetSolver(Solver& solver) {
    Kokkos::deep_copy(solver.A->getLocalMatrixDevice().values, 0.);
    Kokkos::deep_copy(solver.b->getLocalViewDevice(Tpetra::Access::OverwriteAll), 0.);
}

}  // namespace openturbine
