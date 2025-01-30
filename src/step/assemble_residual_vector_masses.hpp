#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/masses/masses.hpp"

namespace openturbine {

namespace masses {

struct AssembleMassesResidual {
    Kokkos::View<double* [6]>::const_type qp_Fi;
    Kokkos::View<double* [6]>::const_type qp_Fg;

    Kokkos::View<double* [6]> residual_vector_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        for (auto i_dof = 0U; i_dof < 6U; ++i_dof) {
            residual_vector_terms(i_elem, i_dof) = qp_Fi(i_elem, i_dof) - qp_Fg(i_elem, i_dof);
        }
    }
};

}  // namespace masses

inline void AssembleResidualVectorMasses(const Masses& masses) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Masses Residual");
    Kokkos::parallel_for(
        "AssembleMassesResidual", masses.num_elems,
        masses::AssembleMassesResidual{masses.qp_Fi, masses.qp_Fg, masses.residual_vector_terms}
    );
}

}  // namespace openturbine
