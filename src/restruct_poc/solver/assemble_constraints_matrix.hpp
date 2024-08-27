#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"
#include "constraints.hpp"
#include "copy_constraints_to_sparse_matrix.hpp"
#include "copy_sparse_values_to_transpose.hpp"

namespace openturbine {
inline void AssembleConstraintsMatrix(Solver& solver, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Matrix");

    if (constraints.num == 0) {
        return;
    }

    {
        auto const_region = Kokkos::Profiling::ScopedRegion("Constraints Matrix"); 
        auto constraint_policy =
            Kokkos::TeamPolicy<>(static_cast<int>(constraints.num), Kokkos::AUTO());
     
        Kokkos::parallel_for(
            "CopyConstraintsToSparseMatrix", constraint_policy,
            CopyConstraintsToSparseMatrix<Solver::CrsMatrixType>{
                constraints.data, solver.B, constraints.gradient_terms}
        );
    }

    {
        auto trans_region = Kokkos::Profiling::ScopedRegion("Transpose Constraints Matrix");
        auto B_num_rows = solver.B.numRows();
        auto constraint_transpose_policy = Kokkos::TeamPolicy<>(B_num_rows, Kokkos::AUTO());
        auto tmp_row_map = Solver::RowPtrType(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "tmp_row_map"),
            solver.B_t.graph.row_map.extent(0)
        );
        Kokkos::deep_copy(tmp_row_map, solver.B_t.graph.row_map);
        Kokkos::parallel_for(
            "CopySparseValuesToTranspose", constraint_transpose_policy,
            CopySparseValuesToTranspose<Solver::CrsMatrixType>{solver.B, tmp_row_map, solver.B_t}
        );
        KokkosSparse::sort_crs_matrix(solver.B_t);
    }

    {
        auto mult_region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Matrix");
        KokkosSparse::spgemm_numeric(
            solver.constraints_spgemm_handle, solver.B, false, solver.T, false,
            solver.constraints_matrix
        );
    }
}
}
