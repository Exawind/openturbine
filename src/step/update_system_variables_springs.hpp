#pragma once

#include <Kokkos_Core.hpp>

#include "assemble_residual_vector_springs.hpp"
#include "assemble_stiffness_matrix_springs.hpp"
#include "elements/springs/springs.hpp"
#include "state/state.hpp"
#include "step_parameters.hpp"
#include "system/springs/calculate_distance_components.hpp"
#include "system/springs/calculate_force_coefficients.hpp"
#include "system/springs/calculate_force_vectors.hpp"
#include "system/springs/calculate_length.hpp"
#include "system/springs/calculate_stiffness_matrix.hpp"
#include "system/springs/update_node_state.hpp"

namespace openturbine {

inline void UpdateSystemVariablesSprings(const Springs& springs, State& state) {
    // Update the displacements from state -> element nodes
    Kokkos::parallel_for(
        springs.num_elems,
        springs::UpdateNodeState{springs.node_state_indices, springs.u1, springs.u2, state.q}
    );

    // Calculate system variables and perform assembly
    Kokkos::parallel_for(
        springs.num_elems,
        KOKKOS_LAMBDA(const size_t i_elem) {
            // Calculate the relative distance vector between the two nodes
            springs::CalculateDistanceComponents{
                i_elem, springs.x0, springs.u1, springs.u2, springs.r
            }();

            // Calculate the current length of the spring
            springs::CalculateLength{i_elem, springs.r, springs.l}();

            // Calculate the force coefficients
            springs::CalculateForceCoefficients{i_elem,    springs.k,  springs.l_ref,
                                                springs.l, springs.c1, springs.c2}();

            // Calculate the force vector
            springs::CalculateForceVectors{i_elem, springs.r, springs.c1, springs.f}();

            // Calculate the stiffness matrix
            springs::CalculateStiffnessMatrix{i_elem,    springs.c1,      springs.c2, springs.r,
                                              springs.l, springs.r_tilde, springs.a}();
        }
    );

    AssembleResidualVectorSprings(springs);
    AssembleStiffnessMatrixSprings(springs);
}

}  // namespace openturbine
