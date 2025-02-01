#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/springs/springs.hpp"

namespace openturbine {

namespace springs {

struct AssembleSpringsStiffness {
    Kokkos::View<double* [3][3]>::const_type a;
    Kokkos::View<double* [2][2][3][3]> stiffness_matrix_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
            for (auto j_dof = 0U; j_dof < 3U; ++j_dof) {
                stiffness_matrix_terms(i_elem, 0, 0, i_dof, j_dof) = a(i_elem, i_dof, j_dof);
            }
        }
        for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
            for (auto j_dof = 0U; j_dof < 3U; ++j_dof) {
                stiffness_matrix_terms(i_elem, 0, 1, i_dof, j_dof) = -a(i_elem, i_dof, j_dof);
            }
        }
        for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
            for (auto j_dof = 0U; j_dof < 3U; ++j_dof) {
                stiffness_matrix_terms(i_elem, 1, 0, i_dof, j_dof) = -a(i_elem, i_dof, j_dof);
            }
        }
        for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
            for (auto j_dof = 0U; j_dof < 3U; ++j_dof) {
                stiffness_matrix_terms(i_elem, 1, 1, i_dof, j_dof) = a(i_elem, i_dof, j_dof);
            }
        }
    }
};

}  // namespace springs

inline void AssembleStiffnessMatrixSprings(const Springs& springs) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Springs Stiffness");
    Kokkos::parallel_for(
        "AssembleSpringsStiffness", springs.num_elems,
        springs::AssembleSpringsStiffness{springs.a, springs.stiffness_matrix_terms}
    );
}

}  // namespace openturbine
