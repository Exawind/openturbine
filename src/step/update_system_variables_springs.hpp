#pragma once

#include <Kokkos_Core.hpp>

#include "assemble_residual_vector_springs.hpp"
#include "assemble_stiffness_matrix_springs.hpp"
#include "elements/springs/springs.hpp"
#include "state/state.hpp"
#include "step_parameters.hpp"
#include "system/springs/calculate_quadrature_point_values.hpp"
#include "system/springs/update_node_state.hpp"

namespace openturbine {

inline void UpdateSystemVariablesSprings(const Springs& springs, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Update System Variables Springs");

    // Update the displacements from state -> element nodes
    Kokkos::parallel_for(
        "springs::UpdateNodeState", springs.num_elems,
        springs::UpdateNodeState{springs.node_state_indices, springs.u1, springs.u2, state.q}
    );

    // Calculate system variables and perform assembly
    Kokkos::parallel_for(
        "Calculate System Variables Springs", springs.num_elems,
        springs::CalculateQuadraturePointValues{
            springs.x0, springs.u1, springs.u2, springs.l_ref, springs.k, springs.r, springs.l,
            springs.c1, springs.c2, springs.f, springs.r_tilde, springs.a
        }
    );

    AssembleResidualVectorSprings(springs);
    AssembleStiffnessMatrixSprings(springs);
}

}  // namespace openturbine
