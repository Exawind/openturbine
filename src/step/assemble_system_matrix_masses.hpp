#pragma once

#include <KokkosBlas.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/elements.hpp"
#include "src/solver/contribute_elements_to_sparse_matrix.hpp"
#include "src/solver/copy_into_sparse_matrix.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemMatrixMasses(Solver& solver, const Masses& masses) {
    const auto num_mass_dofs = 6U;
    const auto row_data_size = Kokkos::View<double*>::shmem_size(num_mass_dofs);
    const auto col_idx_size = Kokkos::View<int*>::shmem_size(num_mass_dofs);
    auto sparse_matrix_policy =
        Kokkos::TeamPolicy<>(static_cast<int>(masses.num_elems), Kokkos::AUTO());

    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerThread(row_data_size + col_idx_size));
    Kokkos::deep_copy(solver.K.values, 0.);
    Kokkos::parallel_for(
        "ContributeElementsToSparseMatrix", sparse_matrix_policy,
        ContributeElementsToSparseMatrix<Solver::CrsMatrixType>{
            masses.num_nodes_per_element, masses.element_freedom_signature,
            masses.element_freedom_table, masses.stiffness_matrix_terms, solver.K
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

    Kokkos::deep_copy(solver.K.values, 0.);
    Kokkos::parallel_for(
        "ContributeElementsToSparseMatrix", sparse_matrix_policy,
        ContributeElementsToSparseMatrix<Solver::CrsMatrixType>{
            masses.num_nodes_per_element, masses.element_freedom_signature,
            masses.element_freedom_table, masses.inertia_matrix_terms, solver.K
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
