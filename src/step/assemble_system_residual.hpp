#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/elements.hpp"
#include "solver/contribute_beams_to_vector.hpp"
#include "solver/contribute_forces_to_vector.hpp"
#include "solver/contribute_masses_to_vector.hpp"
#include "solver/contribute_springs_to_vector.hpp"
#include "solver/solver.hpp"
#include "state/state.hpp"

namespace openturbine {

template <typename DeviceType>
inline void AssembleSystemResidual(
    Solver<DeviceType>& solver, Elements<DeviceType>& elements, State<DeviceType>& state
) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Residual");

    auto forces_vector_policy = Kokkos::RangePolicy<typename DeviceType::execution_space>(
        0, static_cast<int>(state.num_system_nodes)
    );
    auto beams_vector_policy = Kokkos::TeamPolicy<typename DeviceType::execution_space>(
        static_cast<int>(elements.beams.num_elems), Kokkos::AUTO()
    );
    auto masses_vector_policy = Kokkos::RangePolicy<typename DeviceType::execution_space>(
        0, static_cast<int>(elements.masses.num_elems)
    );
    auto springs_vector_policy = Kokkos::RangePolicy<typename DeviceType::execution_space>(
        0, static_cast<int>(elements.springs.num_elems)
    );

    Kokkos::parallel_for(
        "ContributeForcesToVector", forces_vector_policy,
        ContributeForcesToVector<DeviceType>{
            state.node_freedom_allocation_table, state.node_freedom_map_table, state.f, solver.b
        }
    );
    Kokkos::fence();
    Kokkos::parallel_for(
        "ContributeBeamsToVector", beams_vector_policy,
        ContributeBeamsToVector<DeviceType>{
            elements.beams.num_nodes_per_element, elements.beams.element_freedom_table,
            elements.beams.residual_vector_terms, solver.b
        }
    );
    Kokkos::fence();
    Kokkos::parallel_for(
        "ContributeMassesToVector", masses_vector_policy,
        ContributeMassesToVector<DeviceType>{
            elements.masses.element_freedom_table, elements.masses.residual_vector_terms, solver.b
        }
    );
    Kokkos::fence();
    Kokkos::parallel_for(
        "ContributeSpringsToVector", springs_vector_policy,
        ContributeSpringsToVector<DeviceType>{
            elements.springs.element_freedom_table, elements.springs.residual_vector_terms, solver.b
        }
    );
    Kokkos::fence();
}

}  // namespace openturbine
