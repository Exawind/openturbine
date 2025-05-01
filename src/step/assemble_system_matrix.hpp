#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/elements.hpp"
#include "solver/contribute_beams_to_sparse_matrix.hpp"
#include "solver/contribute_masses_to_sparse_matrix.hpp"
#include "solver/contribute_springs_to_sparse_matrix.hpp"
#include "solver/solver.hpp"
#include "step_parameters.hpp"

namespace openturbine {

template <typename DeviceType>
inline void AssembleSystemMatrix(
    StepParameters& parameters, Solver<DeviceType>& solver, Elements& elements
) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Matrix");

    auto beams_sparse_matrix_policy =
        Kokkos::TeamPolicy<>(static_cast<int>(elements.beams.num_elems), Kokkos::AUTO());
    auto masses_sparse_matrix_policy =
        Kokkos::RangePolicy<>(0, static_cast<int>(elements.masses.num_elems));
    auto springs_sparse_matrix_policy =
        Kokkos::RangePolicy<>(0, static_cast<int>(elements.springs.num_elems));

    Kokkos::parallel_for(
        "ContributeBeamsToSparseMatrix", beams_sparse_matrix_policy,
        ContributeBeamsToSparseMatrix<typename Solver<DeviceType>::CrsMatrixType>{
            parameters.conditioner, elements.beams.num_nodes_per_element,
            elements.beams.element_freedom_signature, elements.beams.element_freedom_table,
            elements.beams.system_matrix_terms, solver.A
        }
    );
    Kokkos::parallel_for(
        "ContributeMassesToSparseMatrix", masses_sparse_matrix_policy,
        ContributeMassesToSparseMatrix<typename Solver<DeviceType>::CrsMatrixType>{
            parameters.conditioner, elements.masses.element_freedom_signature,
            elements.masses.element_freedom_table, elements.masses.system_matrix_terms, solver.A
        }
    );
    Kokkos::parallel_for(
        "ContributeSpringsToSparseMatrix", springs_sparse_matrix_policy,
        ContributeSpringsToSparseMatrix<typename Solver<DeviceType>::CrsMatrixType>{
            parameters.conditioner, elements.springs.element_freedom_signature,
            elements.springs.element_freedom_table, elements.springs.stiffness_matrix_terms, solver.A
        }
    );
}

}  // namespace openturbine
