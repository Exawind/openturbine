#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "integrate_matrix.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/masses/masses.hpp"

namespace openturbine {

struct AssembleMasses {
    Kokkos::View<int*>::const_type node_state_indices_;
    View_Nx6x6::const_type node_Muu_;  // Mass matrix in global coordinates
    View_NxN_atomic gbl_M_;            //

    KOKKOS_INLINE_FUNCTION void operator()(const int i_node) const {
        const auto i_gbl_start = node_state_indices_(i_node) * kLieAlgebraComponents;
        for (int m = 0; m < kLieAlgebraComponents; ++m) {
            for (int n = 0; n < kLieAlgebraComponents; ++n) {
                gbl_M_(i_gbl_start + m, i_gbl_start + n) += node_Muu_(i_node, m, n);
            }
        }
    }
};

inline void AssembleMassMatrix(Beams& beams, Masses& masses, View_NxN M) {
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
            beams.elem_indices,
            beams.node_state_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.qp_Muu,
            M,
        }
    );
    Kokkos::parallel_for(
        "AssembleMasses", masses.num_nodes,
        AssembleMasses{
            masses.node_state_indices,
            masses.node_Muu,
            M,
        }
    );
}

}  // namespace openturbine
