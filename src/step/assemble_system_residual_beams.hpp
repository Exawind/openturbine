#pragma once

#include <Kokkos_Core.hpp>

#include "src/elements/beams/beams.hpp"
#include "src/solver/contribute_elements_to_vector.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemResidualBeams(Solver& solver, const Beams& beams) {
    auto vector_policy = Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());
    Kokkos::parallel_for(
        "ContributeElementsToVector", vector_policy,
        ContributeElementsToVector{
            beams.num_nodes_per_element, beams.element_freedom_table, beams.residual_vector_terms,
            solver.R
        }
    );
}

}  // namespace openturbine
