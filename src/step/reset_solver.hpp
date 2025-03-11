#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver/solver.hpp"

namespace openturbine {

inline void ResetSolver(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Reset Solver");
    Kokkos::deep_copy(solver.A->getLocalMatrixDevice().values, 0.);
    Kokkos::deep_copy(solver.b->getLocalViewDevice(Tpetra::Access::OverwriteAll), 0.);
}

}  // namespace openturbine
