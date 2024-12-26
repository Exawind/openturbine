#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/elements.hpp"
#include "src/solver/contribute_beams_to_vector.hpp"
#include "src/solver/contribute_masses_to_vector.hpp"
#include "src/solver/contribute_springs_to_vector.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemResidual(Solver& solver, Elements& elements) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Residual");

    auto beams_vector_policy =
        Kokkos::TeamPolicy<>(static_cast<int>(elements.beams.num_elems), Kokkos::AUTO());
    auto masses_vector_policy =
        Kokkos::RangePolicy<>(0, static_cast<int>(elements.masses.num_elems));
    auto springs_vector_policy =
        Kokkos::RangePolicy<>(0, static_cast<int>(elements.springs.num_elems));

    const auto num_rows = solver.num_system_dofs;
    Kokkos::deep_copy(Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, num_rows)), 0.);

    Kokkos::parallel_for(
        "ContributeBeamsToVector", beams_vector_policy,
        ContributeBeamsToVector{
            elements.beams.num_nodes_per_element, elements.beams.element_freedom_table,
            elements.beams.residual_vector_terms, solver.R
        }
    );
    Kokkos::fence();
    Kokkos::parallel_for(
        "ContributeMassesToVector", masses_vector_policy,
        ContributeMassesToVector{
            elements.masses.element_freedom_table, elements.masses.residual_vector_terms, solver.R
        }
    );
    Kokkos::fence();
    Kokkos::parallel_for(
        "ContributeSpringsToVector", springs_vector_policy,
        ContributeSpringsToVector{
            elements.springs.element_freedom_table, elements.springs.residual_vector_terms, solver.R
        }
    );
    Kokkos::fence();
}

}  // namespace openturbine
