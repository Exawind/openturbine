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
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(static_cast<int>(num_nodes), Kokkos::AUTO());

    Kokkos::parallel_for(
        "CopyTangentToSparseMatrix", sparse_matrix_policy,
        CopyTangentToSparseMatrix<Solver::CrsMatrixType>{
            state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
            state.tangent, solver.T
        }
    );
}

}  // namespace openturbine
