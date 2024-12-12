#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "assemble_system_matrix_beams.hpp"
#include "assemble_system_matrix_masses.hpp"

#include "src/elements/elements.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemMatrix(Solver& solver, Elements& elements) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Matrix");

    if (elements.beams) {
        AssembleSystemMatrixBeams(solver, *elements.beams);
    }

    if (elements.masses) {
        AssembleSystemMatrixMasses(solver, *elements.masses);
    }
}

}  // namespace openturbine
