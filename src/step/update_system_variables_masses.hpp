#pragma once

#include <Kokkos_Core.hpp>

#include "state/state.hpp"
#include "step_parameters.hpp"
#include "system/masses/calculate_quadrature_point_values.hpp"

namespace openturbine {

template <typename DeviceType>
inline void UpdateSystemVariablesMasses(
    const StepParameters& parameters, const Masses<DeviceType>& masses, State& state
) {
    auto region = Kokkos::Profiling::ScopedRegion("Update System Variables Masses");

    Kokkos::parallel_for(
        "masses::CalculateQuadraturePointValues", masses.num_elems,
        masses::CalculateQuadraturePointValues{
            parameters.beta_prime, parameters.gamma_prime, state.q, state.v, state.vd, state.tangent,
            masses.state_indices, masses.gravity, masses.qp_Mstar, masses.node_x0,
            masses.residual_vector_terms, masses.system_matrix_terms
        }
    );
}

}  // namespace openturbine
