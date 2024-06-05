#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_constraint_X0.hpp"
#include "solver.hpp"

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {

inline void InitializeConstraints(Solver& solver, Beams& beams, Masses& masses) {
    auto region = Kokkos::Profiling::ScopedRegion("Initialize Constraints");
    Kokkos::parallel_for(
        "CalculateConstraintX0", solver.num_constraint_nodes,
        CalculateConstraintX0{
            solver.constraints.data,
            beams.node_state_indices,
            beams.node_x0,
            masses.node_state_indices,
            masses.node_x0,
        }
    );
}

}  // namespace openturbine
