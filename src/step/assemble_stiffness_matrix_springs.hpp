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
            for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
                for (auto j_dof = 0U; j_dof < 3U; ++j_dof) {
                    springs.stiffness_matrix_terms(i_elem, 0, 0, i_dof, j_dof) =
                        springs.a(i_elem, i_dof, j_dof);
                    springs.stiffness_matrix_terms(i_elem, 0, 1, i_dof, j_dof) =
                        -springs.a(i_elem, i_dof, j_dof);
                    springs.stiffness_matrix_terms(i_elem, 1, 0, i_dof, j_dof) =
                        -springs.a(i_elem, i_dof, j_dof);
                    springs.stiffness_matrix_terms(i_elem, 1, 1, i_dof, j_dof) =
                        springs.a(i_elem, i_dof, j_dof);
                }
            }
        }
    );
}

}  // namespace openturbine
