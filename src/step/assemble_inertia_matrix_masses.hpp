#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/masses/masses.hpp"

namespace openturbine {

namespace masses {

struct AssembleMassesInertia {
    double beta_prime;
    double gamma_prime;
    Kokkos::View<double* [6][6]>::const_type qp_Muu;
    Kokkos::View<double* [6][6]>::const_type qp_Guu;

    Kokkos::View<double* [6][6]> inertia_matrix_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        for (auto i_dof = 0U; i_dof < 6U; ++i_dof) {
            for (auto j_dof = 0U; j_dof < 6U; ++j_dof) {
                inertia_matrix_terms(i_elem, i_dof, j_dof) =
                    beta_prime * qp_Muu(i_elem, i_dof, j_dof) +
                    gamma_prime * qp_Guu(i_elem, i_dof, j_dof);
            }
        }
    }
};

}  // namespace masses

inline void AssembleInertiaMatrixMasses(
    const Masses& masses, double beta_prime, double gamma_prime
) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Masses Inertia");
    Kokkos::parallel_for(
        "AssembleMassesInertia", masses.num_elems,
        masses::AssembleMassesInertia{
            beta_prime, gamma_prime, masses.qp_Muu, masses.qp_Guu, masses.inertia_matrix_terms
        }
    );
}

}  // namespace openturbine
