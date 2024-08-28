#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/restruct_poc/solver/copy_tangent_to_sparse_matrix.hpp"
#include "src/restruct_poc/solver/solver.hpp"
#include "src/restruct_poc/state/state.hpp"
#include "src/restruct_poc/system/calculate_tangent_operator.hpp"

namespace openturbine {

inline void AssembleTangentOperator(Solver& solver, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Tangent Operator");

    const auto num_rows = solver.num_system_dofs;

    const auto max_row_entries = 6U;
    const auto row_data_size = Kokkos::View<double*>::shmem_size(max_row_entries);
    const auto col_idx_size = Kokkos::View<int*>::shmem_size(max_row_entries);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(static_cast<int>(num_rows), Kokkos::AUTO());

    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for(
        "CopyTangentIntoSparseMatrix", sparse_matrix_policy,
        CopyTangentToSparseMatrix<Solver::CrsMatrixType>{solver.T, state.tangent}
    );
}

}  // namespace openturbine
