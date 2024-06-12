#pragma once

#include <KokkosBlas.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "compute_number_of_non_zeros.hpp"
#include "copy_into_sparse_matrix.hpp"
#include "populate_sparse_indices.hpp"
#include "populate_sparse_row_ptrs.hpp"
#include "solver.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/assemble_elastic_stiffness_matrix.hpp"
#include "src/restruct_poc/system/assemble_gyroscopic_inertia_matrix.hpp"
#include "src/restruct_poc/system/assemble_inertial_stiffness_matrix.hpp"
#include "src/restruct_poc/system/assemble_mass_matrix.hpp"
#include "src/restruct_poc/system/assemble_residual_vector.hpp"
#include "src/restruct_poc/system/calculate_tangent_operator.hpp"

namespace openturbine {

template <typename Subview_NxN, typename Subview_N>
void AssembleSystem(Solver& solver, Beams& beams, Subview_NxN St_11, Subview_N R_system) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System");
    Kokkos::deep_copy(solver.K_dense, 0.);
    Kokkos::parallel_for(
        "TangentOperator", solver.num_system_nodes,
        CalculateTangentOperator{
            solver.h,
            solver.state.q_delta,
            solver.K_dense,
        }
    );

    auto num_rows = solver.num_system_dofs;
    auto num_columns = solver.num_system_dofs;

    auto row_data_size = Kokkos::View<double*>::shmem_size(num_columns);
    auto col_idx_size = Kokkos::View<int*>::shmem_size(num_columns);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(num_rows, Kokkos::AUTO());
    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", sparse_matrix_policy, CopyIntoSparseMatrix{solver.T, solver.K_dense}
    );

    Kokkos::deep_copy(R_system, 0.);
    AssembleResidualVector(beams, R_system);

    Kokkos::deep_copy(solver.K_dense, 0.);
    AssembleElasticStiffnessMatrix(beams, solver.K_dense);
    AssembleInertialStiffnessMatrix(beams, solver.K_dense);

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", sparse_matrix_policy, CopyIntoSparseMatrix{solver.K, solver.K_dense}
    );

    Kokkos::fence();
    auto system_spgemm_handle = Solver::KernelHandle();
    system_spgemm_handle.create_spgemm_handle();
    KokkosSparse::spgemm_symbolic(
        system_spgemm_handle, solver.K, false, solver.T, false, solver.static_system_matrix
    );
    KokkosSparse::spgemm_numeric(
        system_spgemm_handle, solver.K, false, solver.T, false, solver.static_system_matrix
    );

    auto beta_prime = (solver.is_dynamic_solve) ? solver.beta_prime : 0.;
    auto gamma_prime = (solver.is_dynamic_solve) ? solver.gamma_prime : 0.;
    Kokkos::deep_copy(solver.K_dense, 0.);
    AssembleMassMatrix(beams, beta_prime, solver.K_dense);
    AssembleGyroscopicInertiaMatrix(beams, gamma_prime, solver.K_dense);
    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", sparse_matrix_policy, CopyIntoSparseMatrix{solver.K, solver.K_dense}
    );

    Kokkos::fence();
    auto system_spadd_handle = Solver::KernelHandle();
    system_spadd_handle.create_spadd_handle(true);
    KokkosSparse::spadd_symbolic(
        &system_spadd_handle, solver.K, solver.static_system_matrix, solver.system_matrix
    );
    KokkosSparse::spadd_numeric(
        &system_spadd_handle, 1., solver.K, 1., solver.static_system_matrix, solver.system_matrix
    );
}

}  // namespace openturbine
