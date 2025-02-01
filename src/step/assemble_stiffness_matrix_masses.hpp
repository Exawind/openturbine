#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/masses/masses.hpp"

namespace openturbine {

namespace masses {

struct AssembleMassesStiffness {
    Kokkos::View<double* [6][6]>::const_type qp_Kuu;

    Kokkos::View<double* [6][6]> stiffness_matrix_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        for (auto i_dof = 0U; i_dof < 6U; ++i_dof) {
            for (auto j_dof = 0U; j_dof < 6U; ++j_dof) {
                stiffness_matrix_terms(i_elem, i_dof, j_dof) = qp_Kuu(i_elem, i_dof, j_dof);
            }
        }
    }
};
}  // namespace masses

inline void AssembleStiffnessMatrixMasses(const Masses& masses) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Masses Stiffness");
    Kokkos::parallel_for(
        "AssembleMassesStiffness", masses.num_elems,
        masses::AssembleMassesStiffness{masses.qp_Kuu, masses.stiffness_matrix_terms}
    );
}

}  // namespace openturbine
