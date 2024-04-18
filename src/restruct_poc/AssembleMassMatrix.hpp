#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "IntegrateMatrix.hpp"
#include "beams.hpp"

namespace openturbine {

inline void AssembleMassMatrix(Beams& beams, View_NxN M) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Mass Matrix");
    Kokkos::parallel_for(
        "IntegrateMatrix",
        Kokkos::MDRangePolicy{
            {0, 0, 0, 0},
            {beams.num_elems, beams.max_elem_nodes, beams.max_elem_nodes, beams.max_elem_qps}},
        IntegrateMatrix{
            beams.elem_indices,
            beams.node_state_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.qp_Muu,
            M,
        }
    );
}

}  // namespace openturbine
