#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/springs/springs.hpp"

namespace openturbine {

namespace springs {

struct AssembleSpringsResidual {
    Kokkos::View<double* [3]>::const_type f;
    Kokkos::View<double* [2][3]> residual_vector_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        // Apply forces with opposite signs to each node
        for (auto i_dof = 0U; i_dof < 3U; ++i_dof) {
            residual_vector_terms(i_elem, 0, i_dof) = f(i_elem, i_dof);
            residual_vector_terms(i_elem, 1, i_dof) = -f(i_elem, i_dof);
        }
    }
};

}  // namespace springs

inline void AssembleResidualVectorSprings(const Springs& springs) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Springs Residual");
    Kokkos::parallel_for(
        "AssembleSpringsResidual", springs.num_elems,
        springs::AssembleSpringsResidual{springs.f, springs.residual_vector_terms}
    );
}

}  // namespace openturbine
