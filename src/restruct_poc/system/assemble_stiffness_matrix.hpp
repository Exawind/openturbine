#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "integrate_stiffness_matrix.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

inline void AssembleStiffnessMatrix(Beams& beams, Kokkos::View<double***> K) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Stiffness Matrix");
    auto range_policy = Kokkos::TeamPolicy<>(beams.num_elems, Kokkos::AUTO());
    Kokkos::parallel_for(
        "IntegrateStiffnessMatrix", range_policy,
        IntegrateStiffnessMatrix{
            beams.elem_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.shape_deriv,
            beams.qp_Kuu,
            beams.qp_Puu,
            beams.qp_Cuu,
            beams.qp_Ouu,
            beams.qp_Quu,
            K,
        }
    );
}

}  // namespace openturbine
