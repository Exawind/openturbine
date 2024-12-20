#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/springs/springs.hpp"

namespace openturbine {

inline void AssembleStiffnessMatrixSprings(const Springs& springs) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Springs Stiffness");
    Kokkos::parallel_for(
        "AssembleSpringsStiffness", springs.num_elems,
        KOKKOS_LAMBDA(const int i_elem) {
            // Apply stiffness terms with appropriate signs for each node pair
            static constexpr std::array<std::array<double, 2>, 2> node_signs = {{
                {1., -1.},  // Signs for node 1
                {-1., 1.}   // Signs for node 2
            }};
            for (auto i_node = 0U; i_node < 2U; ++i_node) {
                for (auto j_node = 0U; j_node < 2U; ++j_node) {
                    const double sign = node_signs[i_node][j_node];
                    for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
                        for (auto j_dof = 0U; j_dof < 3U; ++j_dof) {
                            springs.stiffness_matrix_terms(i_elem, i_node, j_node, i_dof, j_dof) =
                                sign * springs.a(i_elem, i_dof, j_dof);
                        }
                    }
                }
            }
        }
    );
}

}  // namespace openturbine
