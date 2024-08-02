#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_constraint_residual_gradient.hpp"
#include "compute_number_of_non_zeros.hpp"
#include "contribute_elements_to_sparse_matrix.hpp"
#include "copy_constraints_to_sparse_matrix.hpp"
#include "copy_into_sparse_matrix.hpp"
#include "copy_into_sparse_matrix_transpose.hpp"
#include "copy_sparse_values_to_transpose.hpp"
#include "populate_sparse_indices.hpp"
#include "populate_sparse_row_ptrs.hpp"
#include "solver.hpp"
#include "update_iteration_matrix.hpp"

namespace openturbine {

template <typename Subview_N>
void AssembleConstraints(Solver& solver, Subview_N R_system, Subview_N R_lambda) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Constraints");

    // If no constraints, return
    if (solver.constraints.num == 0) {
        return;
    }

    // Transfer prescribed displacements to host
    solver.constraints.UpdateViews();

    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", solver.constraints.num,
        CalculateConstraintResidualGradient{
            solver.constraints.data, solver.constraints.control, solver.constraints.u,
            solver.state.q, solver.constraints.Phi, solver.constraints.gradient_terms}
    );

    auto constraint_policy =
        Kokkos::TeamPolicy<>(static_cast<int>(solver.constraints.num), Kokkos::AUTO());

    Kokkos::parallel_for(
        "CopyConstraintsToSparseMatrix", constraint_policy,
        CopyConstraintsToSparseMatrix<Solver::CrsMatrixType>{
            solver.constraints.data, solver.B, solver.constraints.gradient_terms}
    );

    Kokkos::fence();

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
        auto resid_region = Kokkos::Profiling::ScopedRegion("Assemble Residual");
        auto R = Solver::ValuesType(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "R_local"), R_system.extent(0)
        );
        Kokkos::deep_copy(R, R_system);
        auto lambda = Solver::ValuesType(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "lambda"), solver.state.lambda.extent(0)
        );
        Kokkos::deep_copy(lambda, solver.state.lambda);
        auto spmv_handle = Solver::SpmvHandle();
        KokkosSparse::spmv(&spmv_handle, "T", 1., solver.B, lambda, 1., R);
        Kokkos::deep_copy(R_system, R);
        Kokkos::deep_copy(R_lambda, solver.constraints.Phi);
    }

    Kokkos::fence();
    {
        auto mult_region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Matrix");
        KokkosSparse::spgemm_numeric(
            solver.constraints_spgemm_handle, solver.B, false, solver.T, false,
            solver.constraints_matrix
        );
    }
}

}  // namespace openturbine
