#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"
#include "src/restruct_poc/system/calculate_tangent_operator.hpp"
#include "copy_tangent_to_sparse_matrix.hpp"

namespace openturbine {

inline void AssembleTangentOperator(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Tangent Operator");
    Kokkos::parallel_for(
        "CalculateTangentOperator", solver.num_system_nodes,
        CalculateTangentOperator{
            solver.h,
            solver.state.q_delta,
            solver.T_dense,
        }
    );

    const auto num_rows = solver.num_system_dofs;

    const auto max_row_entries = 6U;
    const auto row_data_size = Kokkos::View<double*>::shmem_size(max_row_entries);
    const auto col_idx_size = Kokkos::View<int*>::shmem_size(max_row_entries);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(static_cast<int>(num_rows), Kokkos::AUTO());

    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for(
        "CopyTangentIntoSparseMatrix", sparse_matrix_policy,
        CopyTangentToSparseMatrix<Solver::CrsMatrixType>{solver.T, solver.T_dense}
    );

}

}
