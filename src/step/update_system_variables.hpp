#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "update_system_variables_beams.hpp"
#include "update_system_variables_masses.hpp"

#include "src/elements/elements.hpp"
#include "src/state/state.hpp"

namespace openturbine {

inline void UpdateSystemVariables(
    StepParameters& parameters, const Elements& elements, State& state
) {
    auto region = Kokkos::Profiling::ScopedRegion("Update System Variables");

    UpdateSystemVariablesBeams(parameters, elements.beams, state);
    UpdateSystemVariablesMasses(parameters, elements.masses, state);
}

}  // namespace openturbine
