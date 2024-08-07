#pragma once

#include <KokkosBlas.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "compute_number_of_non_zeros.hpp"
#include "contribute_elements_to_sparse_matrix.hpp"
#include "copy_into_sparse_matrix.hpp"
#include "copy_tangent_to_sparse_matrix.hpp"
#include "populate_sparse_indices.hpp"
#include "populate_sparse_row_ptrs.hpp"
#include "solver.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/assemble_inertia_matrix.hpp"
#include "src/restruct_poc/system/assemble_residual_vector.hpp"
#include "src/restruct_poc/system/assemble_stiffness_matrix.hpp"
#include "src/restruct_poc/system/calculate_tangent_operator.hpp"

namespace openturbine {

template <typename Subview_N>
void AssembleSystem(Solver& solver, Beams& beams, Subview_N R_system) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System");
    Kokkos::parallel_for(
        "TangentOperator", solver.num_system_nodes,
        CalculateTangentOperator{
            solver.h,
            solver.state.q_delta,
            solver.T_dense,
        }
    );

    auto num_rows = solver.num_system_dofs;

    auto row_data_size = Kokkos::View<double*>::shmem_size(solver.matrix_terms.extent(2));
    auto col_idx_size = Kokkos::View<int*>::shmem_size(solver.matrix_terms.extent(2));
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(static_cast<int>(num_rows), Kokkos::AUTO());

    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for(
        "CopyTangentIntoSparseMatrix", sparse_matrix_policy,
        CopyTangentToSparseMatrix<Solver::CrsMatrixType>{solver.T, solver.T_dense}
    );

    Kokkos::deep_copy(R_system, 0.);
    AssembleResidualVector(beams, R_system);

    AssembleStiffnessMatrix(beams, solver.matrix_terms);

    Kokkos::parallel_for(
        "ContributeElementsToSparseMatrix", sparse_matrix_policy,
        ContributeElementsToSparseMatrix<Solver::CrsMatrixType>{solver.K, solver.matrix_terms}
    );

    Kokkos::fence();
    {
        auto static_region = Kokkos::Profiling::ScopedRegion("Assemble Static System Matrix");
        KokkosSparse::spgemm_numeric(
            solver.system_spgemm_handle, solver.K, false, solver.T, false,
            solver.static_system_matrix
        );
    }

    auto beta_prime = (solver.is_dynamic_solve) ? solver.beta_prime : 0.;
    auto gamma_prime = (solver.is_dynamic_solve) ? solver.gamma_prime : 0.;
    AssembleInertiaMatrix(beams, beta_prime, gamma_prime, solver.matrix_terms);
    Kokkos::parallel_for(
        "ContributeElementsToSparseMatrix", sparse_matrix_policy,
        ContributeElementsToSparseMatrix<Solver::CrsMatrixType>{solver.K, solver.matrix_terms}
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
