#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/springs/springs.hpp"

namespace openturbine {

inline void AssembleResidualVectorSprings(const Springs& springs) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Springs Residual");
    Kokkos::parallel_for(
        "AssembleSpringsResidual", springs.num_elems,
        KOKKOS_LAMBDA(const int i_elem) {
            // Apply forces with opposite signs to each node
            for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
                springs.residual_vector_terms(i_elem, 0, i_dof) = springs.f(i_elem, i_dof);
                springs.residual_vector_terms(i_elem, 1, i_dof) = -springs.f(i_elem, i_dof);
            }
        }
    );
}

}  // namespace openturbine
