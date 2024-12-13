#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/solver/copy_tangent_to_sparse_matrix.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/system/calculate_tangent_operator.hpp"

namespace openturbine {

inline void AssembleTangentOperator(Solver& solver, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Tangent Operator");

    const auto num_nodes = state.num_system_nodes;
    const auto max_row_entries = 6U;
    const auto row_data_size = Kokkos::View<double*>::shmem_size(max_row_entries);
    const auto col_idx_size = Kokkos::View<int*>::shmem_size(max_row_entries);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(static_cast<int>(num_nodes), Kokkos::AUTO());

    // TODO: Hack to avoid the segfault. Why do we need to increase the scratch size here?
    sparse_matrix_policy.set_scratch_size(
        1, Kokkos::PerThread(6 * row_data_size + 6 * col_idx_size)
    );

    Kokkos::parallel_for(
        "CopyTangentToSparseMatris", sparse_matrix_policy,
        CopyTangentToSparseMatrix<Solver::CrsMatrixType>{
            state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
            state.tangent, solver.T
        }
    );
}

}  // namespace openturbine
