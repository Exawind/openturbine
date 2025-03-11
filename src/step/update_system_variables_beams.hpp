#pragma once

#include <Kokkos_Core.hpp>

#include "elements/beams/beams.hpp"
#include "state/state.hpp"
#include "step_parameters.hpp"
#include "system/beams/calculate_quadrature_point_values.hpp"

namespace openturbine {

inline void UpdateSystemVariablesBeams(
    StepParameters& parameters, const Beams& beams, State& state
) {
    const auto num_nodes = beams.max_elem_nodes;
    const auto num_qps = beams.max_elem_qps;

    auto range_policy = Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());
    const auto shape_size = Kokkos::View<double**>::shmem_size(num_nodes, num_qps);
    const auto weight_size = Kokkos::View<double*>::shmem_size(num_qps);
    const auto node_variable_size = Kokkos::View<double* [7]>::shmem_size(num_nodes);
    const auto qp_variable_size = Kokkos::View<double* [6]>::shmem_size(num_qps);
    const auto qp_matrix_size = Kokkos::View<double* [6][6]>::shmem_size(num_qps);
    const auto system_matrix_size = Kokkos::View<double** [6][6]>::shmem_size(num_nodes, num_nodes);
    auto smem = 2 * shape_size + 2 * weight_size + 4 * node_variable_size + 5 * qp_variable_size +
                7 * qp_matrix_size + 2 * system_matrix_size;
    range_policy.set_scratch_size(1, Kokkos::PerTeam(smem));

    Kokkos::parallel_for(
        "CalculateQuadraturePointValues", range_policy,
        beams::CalculateQuadraturePointValues{
            parameters.beta_prime,
            parameters.gamma_prime,
            state.q,
            state.v,
            state.vd,
            state.tangent,
            beams.node_state_indices,
            beams.num_nodes_per_element,
            beams.num_qps_per_element,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.shape_deriv,
            beams.gravity,
            beams.node_FX,
            beams.qp_r0,
            beams.qp_x0,
            beams.qp_x0_prime,
            beams.qp_Mstar,
            beams.qp_Cstar,
            beams.qp_Fe,
            beams.residual_vector_terms,
            beams.system_matrix_terms
        }
    );
}

}  // namespace openturbine
