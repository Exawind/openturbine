#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "integrate_matrix.hpp"

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {

inline void AssembleMassMatrix(Beams& beams, double beta_prime, View_NxN M) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Mass Matrix");
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
            beams.elem_indices, beams.node_state_indices, beams.qp_weight, beams.qp_jacobian,
            beams.shape_interp, beams.qp_Muu, M, beta_prime}
    );
}

}  // namespace openturbine
