#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "assemble_system_residual_beams.hpp"
#include "assemble_system_residual_masses.hpp"

#include "src/elements/elements.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemResidual(Solver& solver, Elements& elements) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Residual");

    const auto num_rows = solver.num_system_dofs;
    Kokkos::deep_copy(Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, num_rows)), 0.);

    if (elements.beams) {
        AssembleSystemResidualBeams(solver, *elements.beams);
    }

    if (elements.masses) {
        AssembleSystemResidualMasses(solver, *elements.masses);
    }
}

}  // namespace openturbine
