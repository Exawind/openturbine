#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/springs/springs.hpp"

namespace openturbine {

inline void AssembleResidualVectorSprings(const Springs& springs) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Springs Residual");
    Kokkos::parallel_for(
        "AssembleSpringsResidual", springs.num_elems,
        KOKKOS_LAMBDA(const int i_elem) {
            // Apply forces with opposite signs to each node
            static constexpr std::array<double, 2> node_signs = {1., -1.};
            for (auto i_node = 0U; i_node < 2U; ++i_node) {
                for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
                    springs.residual_vector_terms(i_elem, i_node, i_dof) =
                        node_signs[i_node] * springs.f(i_elem, i_dof);
                }
            }
        }
    );
}

}  // namespace openturbine
