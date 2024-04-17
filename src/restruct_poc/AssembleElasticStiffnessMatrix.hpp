#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "IntegrateElasticStiffnessMatrix.hpp"
#include "beams.hpp"
#include "types.hpp"

namespace openturbine {

inline void AssembleElasticStiffnessMatrix(Beams& beams, View_NxN K) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Elastic Stiffness Matrix");
    Kokkos::parallel_for(
        "IntegrateElasticStiffnessMatrix",
        Kokkos::MDRangePolicy{
            {0, 0, 0, 0},
            {beams.num_elems, beams.max_elem_nodes, beams.max_elem_nodes, beams.max_elem_qps}},
        IntegrateElasticStiffnessMatrix{
            beams.elem_indices,
            beams.node_state_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.shape_deriv,
            beams.qp_Puu,
            beams.qp_Cuu,
            beams.qp_Ouu,
            beams.qp_Quu,
            K,
        }
    );
}

}  // namespace openturbine