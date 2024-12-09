#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "assemble_system_matrix_beams.hpp"

#include "src/elements/elements.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemMatrix(Solver& solver, Elements& elements) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Matrix");

    // Assemble Beams matrix
    if (elements.beams) {
        AssembleSystemMatrixBeams(solver, *elements.beams);
    }

    // TODO: Assemble Masses matrix
}

}  // namespace openturbine
