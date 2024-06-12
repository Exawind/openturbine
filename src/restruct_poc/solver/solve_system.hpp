#pragma once

#include <KokkosBlas.hpp>
#include <KokkosLapack_gesv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "fill_unshifted_row_ptrs.hpp"
#include "condition_system.hpp"
#include "solver.hpp"

namespace openturbine {

inline void SolveSystem(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Solve System");
    using CrsMatrixType = typename Solver::CrsMatrixType;
    auto num_dofs = solver.num_dofs;
    auto num_system_dofs = solver.num_system_dofs;
    auto system_matrix = solver.system_matrix;
    auto system_matrix_full_row_ptrs = Kokkos::View<int*>("system_matrix_full_row_ptrs", num_dofs+1);
    Kokkos::parallel_for("FillUnshiftedRowPtrs", num_dofs+1, FillUnshiftedRowPtrs{system_matrix_full_row_ptrs, num_system_dofs, system_matrix.graph.row_map});
    
    auto system_matrix_full = CrsMatrixType("system_matrix_full", num_dofs, num_dofs, system_matrix.nnz(), system_matrix.values, system_matrix_full_row_ptrs, system_matrix.graph.entries);

    auto constraints_matrix = solver.constraints_matrix;
    auto constraints_matrix_full_row_ptrs = Kokkos::View<int*>("constraints_matrix_full_row_ptrs", num_dofs+1);
    Kokkos::parallel_for("FillConstraintsMatrixFullRowPtrs", num_dofs+1, KOKKOS_LAMBDA(int i) {
        if(i > num_system_dofs) {
            constraints_matrix_full_row_ptrs(i) = constraints_matrix.graph.row_map(i-num_system_dofs);
        }
    });
    auto constraints_matrix_full = CrsMatrixType("constraints_matrix_full", num_dofs, num_dofs, constraints_matrix.nnz(), constraints_matrix.values, constraints_matrix_full_row_ptrs, constraints_matrix.graph.entries);

    auto transpose_matrix = solver.B_t;
    auto transpose_matrix_full_row_ptrs = Kokkos::View<int*>("transpose_matrix_full_row_ptrs", num_dofs+1);
    Kokkos::parallel_for("FillUnshiftedRowPtrs", num_dofs+1, FillUnshiftedRowPtrs{transpose_matrix_full_row_ptrs, num_system_dofs, transpose_matrix.graph.row_map});

    auto transpose_matrix_full_indices = Kokkos::View<int*>("transpose_matrix_full_indices", transpose_matrix.nnz());
    Kokkos::parallel_for("fullTransposeMatrixFullIndices", transpose_matrix.nnz(), KOKKOS_LAMBDA(int i) {
        transpose_matrix_full_indices(i) = transpose_matrix.graph.entries(i) + num_system_dofs;
    });
    auto transpose_matrix_full = CrsMatrixType("transpose_matrix_full", num_dofs, num_dofs, transpose_matrix.nnz(), transpose_matrix.values, transpose_matrix_full_row_ptrs, transpose_matrix_full_indices);

    Kokkos::fence();
    auto spc_handle = Solver::KernelHandle();
    spc_handle.create_spadd_handle(true);
    KokkosSparse::spadd_symbolic(&spc_handle, system_matrix_full, constraints_matrix_full, solver.system_plus_constraints);
    KokkosSparse::spadd_numeric(&spc_handle, solver.conditioner, system_matrix_full, 1., constraints_matrix_full, solver.system_plus_constraints);

    auto system_handle = Solver::KernelHandle();
    system_handle.create_spadd_handle(true);
    KokkosSparse::spadd_symbolic(&system_handle, solver.system_plus_constraints, transpose_matrix_full, solver.full_matrix);
    KokkosSparse::spadd_numeric(&system_handle, 1., solver.system_plus_constraints, 1., transpose_matrix_full, solver.full_matrix);

    auto St = solver.St;
    auto full_matrix = solver.full_matrix;
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(num_dofs, Kokkos::AUTO());
    Kokkos::parallel_for("Copy into St", sparse_matrix_policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
            auto i = member.league_rank();
            auto row = full_matrix.row(i);
            auto row_map = full_matrix.graph.row_map;
            auto cols = full_matrix.graph.entries;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
                St(i, cols(row_map(i) + entry)) = row.value(entry);
            });
    });

    Kokkos::parallel_for(
        "ConditionR", solver.num_system_dofs,
        ConditionR{
            solver.R,
            solver.conditioner,
        }
    );

    KokkosBlas::axpby(-1.0, solver.R, 0.0, solver.x);
    auto x = Kokkos::View<double*, Kokkos::LayoutLeft>(solver.x);
    if constexpr (std::is_same_v<decltype(solver.St)::array_layout, Kokkos::LayoutLeft>) {
        KokkosLapack::gesv(solver.St, x, solver.IPIV);
    } else {
        Kokkos::deep_copy(solver.St_left, solver.St);
        KokkosLapack::gesv(solver.St_left, x, solver.IPIV);
    }

    Kokkos::parallel_for(
        "UnconditionSolution", solver.num_dofs,
        UnconditionSolution{
            solver.num_system_dofs,
            solver.conditioner,
            solver.x,
        }
    );
}

}  // namespace openturbine
