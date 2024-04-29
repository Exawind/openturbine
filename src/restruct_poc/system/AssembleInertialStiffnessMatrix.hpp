#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "IntegrateMatrix.hpp"

#include "src/restruct_poc/beams/Beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

inline void AssembleInertialStiffnessMatrix(Beams& beams, View_NxN K) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Inertial Stiffness Matrix");
    auto range_policy = std::invoke([&]() {
        if constexpr (std::is_same_v<
                          Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
            return Kokkos::MDRangePolicy{
                {0, 0, 0}, {beams.num_elems, beams.max_elem_nodes, beams.max_elem_nodes}};
        } else {
            return Kokkos::MDRangePolicy{
                {0, 0, 0, 0},
                {beams.num_elems, beams.max_elem_nodes, beams.max_elem_nodes, beams.max_elem_qps}};
        }
    });
    Kokkos::parallel_for(
        "IntegrateMatrix", range_policy,
        IntegrateMatrix{
            beams.elem_indices,
            beams.node_state_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.qp_Kuu,
            K,
        }
    );
}

}  // namespace openturbine
