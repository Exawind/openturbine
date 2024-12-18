#pragma once

#include <Kokkos_Core.hpp>

#include "src/elements/masses/masses.hpp"
#include "src/solver/contribute_elements_to_vector.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemResidualMasses(Solver& solver, const Masses& masses) {
    auto vector_policy = Kokkos::TeamPolicy<>(static_cast<int>(masses.num_elems), Kokkos::AUTO());
    Kokkos::parallel_for(
        "ContributeElementsToVector", vector_policy,
        ContributeElementsToVector{
            masses.num_nodes_per_element, masses.element_freedom_table, masses.residual_vector_terms,
            solver.R
        }
    );
}

}  // namespace openturbine
