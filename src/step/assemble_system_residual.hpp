#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/beams/beams.hpp"
#include "src/solver/contribute_elements_to_vector.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemResidual(Solver& solver, Beams& beams) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Residual");

    const auto num_rows = solver.num_system_dofs;

    Kokkos::deep_copy(Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, num_rows)), 0.);
    auto vector_policy = Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());
    Kokkos::parallel_for(
        "ContributeElementsToVector", vector_policy,
        ContributeElementsToVector{
            beams.num_nodes_per_element, beams.node_state_indices, beams.residual_vector_terms,
            solver.R
        }
    );
}

}  // namespace openturbine
