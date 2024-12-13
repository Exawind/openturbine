#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/masses/masses.hpp"

namespace openturbine {

inline void AssembleResidualVectorMasses(const Masses& masses) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Masses Residual");
    Kokkos::parallel_for(
        "AssembleMassesResidual", masses.num_elems,
        KOKKOS_LAMBDA(const int i_elem) {
            // Add inertial (Fi) to and subtract gravity (Fg) forces from residual
            for (auto i_dof = 0U; i_dof < 6U; ++i_dof) {
                masses.residual_vector_terms(i_elem, 0U, i_dof) +=
                    masses.Fi(i_elem, 0U, i_dof) - masses.Fg(i_elem, 0U, i_dof);
            }
        }
    );
}

}  // namespace openturbine
