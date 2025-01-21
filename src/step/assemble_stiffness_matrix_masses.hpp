#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/masses/masses.hpp"

namespace openturbine {

inline void AssembleStiffnessMatrixMasses(const Masses& masses) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Masses Stiffness");
    Kokkos::parallel_for(
        "AssembleMassesStiffness", masses.num_elems,
        KOKKOS_LAMBDA(const int i_elem) {
            // Add stiffness terms to global stiffness matrix
            for (auto i_dof = 0U; i_dof < 6U; ++i_dof) {
                for (auto j_dof = 0U; j_dof < 6U; ++j_dof) {
                    masses.stiffness_matrix_terms(i_elem, i_dof, j_dof) =
                        masses.qp_Kuu(i_elem, i_dof, j_dof);
                }
            }
        }
    );
}

}  // namespace openturbine
