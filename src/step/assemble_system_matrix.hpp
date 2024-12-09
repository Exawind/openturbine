#pragma once

#include <KokkosBlas.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/beams/beams.hpp"
#include "src/solver/contribute_elements_to_sparse_matrix.hpp"
#include "src/solver/copy_into_sparse_matrix.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleSystemMatrix(Solver& solver, Beams& beams) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System Matrix");

    auto sparse_matrix_policy =
        Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());

    Kokkos::deep_copy(solver.K.values, 0.);
    Kokkos::parallel_for(
        "ContributeElementsToSparseMatrix", sparse_matrix_policy,
        ContributeElementsToSparseMatrix<Solver::CrsMatrixType>{
            beams.num_nodes_per_element, beams.element_freedom_signature,
            beams.element_freedom_table, beams.stiffness_matrix_terms, solver.K
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
            beams.num_nodes_per_element, beams.element_freedom_signature,
            beams.element_freedom_table, beams.inertia_matrix_terms, solver.K
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
