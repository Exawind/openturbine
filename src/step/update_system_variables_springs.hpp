#pragma once

#include <Kokkos_Core.hpp>

#include "assemble_residual_vector_springs.hpp"
#include "assemble_stiffness_matrix_springs.hpp"
#include "step_parameters.hpp"

#include "src/elements/springs/springs.hpp"
#include "src/state/state.hpp"
#include "src/system/springs/calculate_distance_components.hpp"
#include "src/system/springs/calculate_force_coefficients.hpp"
#include "src/system/springs/calculate_force_vectors.hpp"
#include "src/system/springs/calculate_length.hpp"
#include "src/system/springs/calculate_stiffness_matrix.hpp"
#include "src/system/springs/update_node_state.hpp"

namespace openturbine {

inline void UpdateSystemVariablesSprings(const Springs& springs, State& state) {
    // Update the displacements from state -> element nodes
    Kokkos::parallel_for(
        springs.num_elems,
        springs::UpdateNodeState{springs.node_state_indices, springs.u1, springs.u2, state.q}
    );

    // Calculate system variables for Spring elements
    Kokkos::parallel_for(
        springs.num_elems,
        KOKKOS_LAMBDA(const size_t i_elem) {
            CalculateDistanceComponents{
                springs.x0, springs.u1, springs.u2, springs.r
            }(static_cast<int>(i_elem));

            CalculateLength{springs.r, springs.l}(static_cast<int>(i_elem));

            CalculateForceCoefficients{
                springs.k, springs.l_ref, springs.l, springs.c1, springs.c2
            }(static_cast<int>(i_elem));

            CalculateForceVectors{springs.r, springs.c1, springs.f}(static_cast<int>(i_elem));

            CalculateStiffnessMatrix{
                springs.c1, springs.c2, springs.r, springs.l, springs.r_tilde, springs.a
            }(static_cast<int>(i_elem));
        }
    );

    AssembleResidualVectorSprings(springs);
    AssembleStiffnessMatrixSprings(springs);
}

}  // namespace openturbine
