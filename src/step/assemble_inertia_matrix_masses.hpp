#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/masses/masses.hpp"

namespace openturbine {

inline void AssembleInertiaMatrixMasses(
    const Masses& masses, double beta_prime, double gamma_prime
) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Masses Inertia");
    Kokkos::parallel_for(
        "AssembleMassesInertia", masses.num_elems,
        KOKKOS_LAMBDA(const int i_elem) {
            // Add inertial terms to global inertia matrix
            // Combines mass matrix (Muu) and gyroscopic terms (Guu)
            for (auto i_dof = 0U; i_dof < 6U; ++i_dof) {
                for (auto j_dof = 0U; j_dof < 6U; ++j_dof) {
                    masses.inertia_matrix_terms(i_elem, 0U, 0U, i_dof, j_dof) +=
                        beta_prime * masses.Muu(i_elem, 0U, i_dof, j_dof) +
                        gamma_prime * masses.Guu(i_elem, 0U, i_dof, j_dof);
                }
            }
        }
    );
}

}  // namespace openturbine
