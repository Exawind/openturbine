#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/elements.hpp"
#include "src/solver/contribute_beams_to_sparse_matrix.hpp"
#include "src/solver/contribute_masses_to_sparse_matrix.hpp"
#include "src/solver/contribute_springs_to_sparse_matrix.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemMatrix(Solver& solver, Elements& elements) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Matrix");

    auto beams_sparse_matrix_policy =
        Kokkos::TeamPolicy<>(static_cast<int>(elements.beams.num_elems), Kokkos::AUTO());
    auto masses_sparse_matrix_policy =
        Kokkos::RangePolicy<>(0, static_cast<int>(elements.masses.num_elems));
    auto springs_sparse_matrix_policy =
        Kokkos::RangePolicy<>(0, static_cast<int>(elements.springs.num_elems));

    // Assemble stiffness matrix by contributing beam, mass, and spring stiffness matrices to the
    // global sparse matrix
    Kokkos::deep_copy(solver.K.values, 0.);
    Kokkos::parallel_for(
        "ContributeBeamsToSparseMatrix", beams_sparse_matrix_policy,
        ContributeBeamsToSparseMatrix<Solver::CrsMatrixType>{
            elements.beams.num_nodes_per_element, elements.beams.element_freedom_signature,
            elements.beams.element_freedom_table, elements.beams.stiffness_matrix_terms, solver.K
        }
    );
    Kokkos::fence();
    Kokkos::parallel_for(
        "ContributeMassesToSparseMatrix", masses_sparse_matrix_policy,
        ContributeMassesToSparseMatrix<Solver::CrsMatrixType>{
            elements.masses.element_freedom_signature, elements.masses.element_freedom_table,
            elements.masses.stiffness_matrix_terms, solver.K
        }
    );
    Kokkos::fence();
    Kokkos::parallel_for(
        "ContributeSpringsToSparseMatrix", springs_sparse_matrix_policy,
        ContributeSpringsToSparseMatrix<Solver::CrsMatrixType>{
            elements.springs.element_freedom_signature, elements.springs.element_freedom_table,
            elements.springs.stiffness_matrix_terms, solver.K
        }
    );

    Kokkos::fence();
    {
        auto static_region = Kokkos::Profiling::ScopedRegion("Assemble Static System Matrix");
        KokkosSparse::spgemm_numeric(
            solver.system_spgemm_handle, solver.K, false, solver.T, false,
            solver.static_system_matrix
        );
    }

    // Assemble inertia matrix by contributing beam and mass inertia matrices to the global
    // sparse matrix
    Kokkos::deep_copy(solver.K.values, 0.);
    Kokkos::parallel_for(
        "ContributeBeamsToSparseMatrix", beams_sparse_matrix_policy,
        ContributeBeamsToSparseMatrix<Solver::CrsMatrixType>{
            elements.beams.num_nodes_per_element, elements.beams.element_freedom_signature,
            elements.beams.element_freedom_table, elements.beams.inertia_matrix_terms, solver.K
        }
    );
    Kokkos::fence();
    Kokkos::parallel_for(
        "ContributeMassesToSparseMatrix", masses_sparse_matrix_policy,
        ContributeMassesToSparseMatrix<Solver::CrsMatrixType>{
            elements.masses.element_freedom_signature, elements.masses.element_freedom_table,
            elements.masses.inertia_matrix_terms, solver.K
        }
    );

    Kokkos::fence();
    {
        auto system_region = Kokkos::Profiling::ScopedRegion("Assemble System Matrix");
        KokkosSparse::spadd_numeric(
            &solver.system_spadd_handle, 1., solver.K, 1., solver.static_system_matrix,
            solver.system_matrix
        );
    }
}

}  // namespace openturbine
