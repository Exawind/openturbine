#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/elements.hpp"
#include "solver/contribute_beams_to_sparse_matrix.hpp"
#include "solver/contribute_masses_to_sparse_matrix.hpp"
#include "solver/contribute_springs_to_sparse_matrix.hpp"
#include "solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemMatrix(Solver& solver, Elements& elements) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Matrix");

    auto beams_sparse_matrix_policy =
        Kokkos::TeamPolicy<>(static_cast<int>(elements.beams.num_elems), Kokkos::AUTO());
    auto masses_sparse_matrix_policy =
        Kokkos::RangePolicy<>(0, static_cast<int>(elements.masses.num_elems));
    auto springs_sparse_matrix_policy =
        Kokkos::RangePolicy<>(0, static_cast<int>(elements.springs.num_elems));

    Kokkos::deep_copy(solver.system_matrix.values, 0.);
    Kokkos::parallel_for(
        "ContributeBeamsToSparseMatrix", beams_sparse_matrix_policy,
        ContributeBeamsToSparseMatrix<Solver::CrsMatrixType>{
            elements.beams.num_nodes_per_element, elements.beams.element_freedom_signature,
            elements.beams.element_freedom_table, elements.beams.system_matrix_terms,
            solver.system_matrix
        }
    );
    Kokkos::parallel_for(
        "ContributeMassesToSparseMatrix", masses_sparse_matrix_policy,
        ContributeMassesToSparseMatrix<Solver::CrsMatrixType>{
            elements.masses.element_freedom_signature, elements.masses.element_freedom_table,
            elements.masses.system_matrix_terms, solver.system_matrix
        }
    );
    Kokkos::parallel_for(
        "ContributeSpringsToSparseMatrix", springs_sparse_matrix_policy,
        ContributeSpringsToSparseMatrix<Solver::CrsMatrixType>{
            elements.springs.element_freedom_signature, elements.springs.element_freedom_table,
            elements.springs.stiffness_matrix_terms, solver.system_matrix
        }
    );
}

}  // namespace openturbine
