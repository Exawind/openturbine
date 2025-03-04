#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/constraints.hpp"
#include "solver/copy_constraints_to_sparse_matrix.hpp"
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
                constraints.target_node_col_range, solver.B, constraints.base_gradient_terms,
                constraints.target_gradient_terms
            }
        );
    }

    {
        auto trans_region = Kokkos::Profiling::ScopedRegion("Transpose Constraints Matrix");
        auto B_num_rows = solver.B.numRows();
        auto constraint_transpose_policy = Kokkos::TeamPolicy<>(B_num_rows, Kokkos::AUTO());
        auto tmp_row_map = Solver::CrsMatrixType::row_map_type::non_const_type(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "tmp_row_map"),
            solver.B_t.graph.row_map.extent(0)
        );
        Kokkos::deep_copy(tmp_row_map, solver.B_t.graph.row_map);
        Kokkos::parallel_for(
            "CopySparseValuesToTranspose", constraint_transpose_policy,
            CopySparseValuesToTranspose<Solver::CrsMatrixType>{solver.B, tmp_row_map, solver.B_t}
        );
        {
            auto sort_region = Kokkos::Profiling::ScopedRegion("Sort");
            KokkosSparse::sort_crs_matrix(solver.B_t);
        }
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
