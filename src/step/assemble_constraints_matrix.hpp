#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/constraints.hpp"
#include "solver/copy_constraints_to_sparse_matrix.hpp"
#include "solver/copy_constraints_transpose_to_sparse_matrix.hpp"
#include "solver/copy_sparse_values_to_transpose.hpp"
#include "solver/solver.hpp"

namespace openturbine {
inline void AssembleConstraintsMatrix(Solver& solver, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Matrix");

    if (constraints.num_constraints == 0) {
        return;
    }

    {
        auto const_region = Kokkos::Profiling::ScopedRegion("Constraints Matrix");
        auto constraint_policy =
            Kokkos::TeamPolicy<>(static_cast<int>(constraints.num_constraints), Kokkos::AUTO());

        Kokkos::parallel_for(
            "CopyConstraintsToSparseMatrix", constraint_policy,
            CopyConstraintsToSparseMatrix<Solver::CrsMatrixType>{
                constraints.row_range, constraints.base_node_col_range,
                constraints.target_node_col_range, constraints.base_gradient_terms,
                constraints.target_gradient_terms, solver.B
            }
        );
    }

    {
        auto trans_region = Kokkos::Profiling::ScopedRegion("Transpose Constraints Matrix");
        auto constraint_policy =
            Kokkos::TeamPolicy<>(static_cast<int>(constraints.num_constraints), Kokkos::AUTO());

        Kokkos::parallel_for(
            "CopyConstraintsTransposeToSparseMatrix", constraint_policy,
            CopyConstraintsTransposeToSparseMatrix<Solver::CrsMatrixType>{
                constraints.row_range, constraints.base_node_col_range,
                constraints.target_node_col_range, constraints.base_node_freedom_signature, constraints.target_node_freedom_signature, constraints.base_node_freedom_table, constraints.target_node_freedom_table, constraints.base_gradient_transpose_terms,
                constraints.target_gradient_transpose_terms, solver.B_t
            }
        );

    }

    {
        auto mult_region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Matrix");
        KokkosSparse::spgemm_numeric(
            solver.constraints_spgemm_handle, solver.B, false, solver.T, false,
            solver.constraints_matrix
        );
    }
}
}  // namespace openturbine
