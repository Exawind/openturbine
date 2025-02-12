#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/springs/springs.hpp"
#include "state/state.hpp"
#include "system/springs/calculate_quadrature_point_values.hpp"

namespace openturbine {

inline void UpdateSystemVariablesSprings(const Springs& springs, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Update System Variables Springs");

    // Calculate system variables and perform assembly
    Kokkos::parallel_for(
        "Calculate System Variables Springs", springs.num_elems,
        springs::CalculateQuadraturePointValues{
            state.q, springs.node_state_indices, springs.x0, springs.l_ref, springs.k,
            springs.residual_vector_terms, springs.stiffness_matrix_terms}
    );
}

}  // namespace openturbine
