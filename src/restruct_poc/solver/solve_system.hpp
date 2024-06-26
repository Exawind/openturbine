#pragma once

#include <KokkosBlas.hpp>
#include <KokkosLapack_gesv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <Teuchos_RCP.hpp>
#include <Amesos2.hpp>

#include "condition_system.hpp"
#include "fill_unshifted_row_ptrs.hpp"
#include "solver.hpp"

namespace openturbine {

inline void SolveSystem(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Solve System");

    using CrsMatrixType = typename Solver::CrsMatrixType;
    auto num_dofs = solver.num_dofs;
    auto num_system_dofs = solver.num_system_dofs;
    auto system_matrix = solver.system_matrix;
    auto system_matrix_full_row_ptrs =
        Solver::RowPtrType("system_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::parallel_for(
        "FillUnshiftedRowPtrs", num_dofs + 1,
        FillUnshiftedRowPtrs<CrsMatrixType::size_type, CrsMatrixType::size_type>{
            system_matrix_full_row_ptrs, num_system_dofs, system_matrix.graph.row_map}
    );

    auto system_matrix_full = CrsMatrixType(
        "system_matrix_full", num_dofs, num_dofs, system_matrix.nnz(), system_matrix.values,
        system_matrix_full_row_ptrs, system_matrix.graph.entries
    );

    auto constraints_matrix = solver.constraints_matrix;
    auto constraints_matrix_full_row_ptrs =
        Solver::RowPtrType("constraints_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::parallel_for(
        "FillConstraintsMatrixFullRowPtrs", num_dofs + 1,
        KOKKOS_LAMBDA(int i) {
            if (i > num_system_dofs) {
                constraints_matrix_full_row_ptrs(i) =
                    constraints_matrix.graph.row_map(i - num_system_dofs);
            }
        }
    );
    auto constraints_matrix_full = CrsMatrixType(
        "constraints_matrix_full", num_dofs, num_dofs, constraints_matrix.nnz(),
        constraints_matrix.values, constraints_matrix_full_row_ptrs, constraints_matrix.graph.entries
    );

    auto transpose_matrix = solver.B_t;
    auto transpose_matrix_full_row_ptrs =
        Solver::RowPtrType("transpose_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::parallel_for(
        "FillUnshiftedRowPtrs", num_dofs + 1,
        FillUnshiftedRowPtrs<CrsMatrixType::size_type, CrsMatrixType::size_type>{
            transpose_matrix_full_row_ptrs, num_system_dofs, transpose_matrix.graph.row_map}
    );

    auto transpose_matrix_full_indices =
        Solver::IndicesType("transpose_matrix_full_indices", transpose_matrix.nnz());
    Kokkos::parallel_for(
        "fullTransposeMatrixFullIndices", transpose_matrix.nnz(),
        KOKKOS_LAMBDA(int i) {
            transpose_matrix_full_indices(i) = transpose_matrix.graph.entries(i) + num_system_dofs;
        }
    );
    auto transpose_matrix_full = CrsMatrixType(
        "transpose_matrix_full", num_dofs, num_dofs, transpose_matrix.nnz(), transpose_matrix.values,
        transpose_matrix_full_row_ptrs, transpose_matrix_full_indices
    );

    Kokkos::fence();
    auto spc_handle = Solver::KernelHandle();
    spc_handle.create_spadd_handle(true);
    KokkosSparse::spadd_symbolic(
        &spc_handle, system_matrix_full, constraints_matrix_full, solver.system_plus_constraints
    );
    KokkosSparse::spadd_numeric(
        &spc_handle, solver.conditioner, system_matrix_full, 1., constraints_matrix_full,
        solver.system_plus_constraints
    );

    auto system_handle = Solver::KernelHandle();
    system_handle.create_spadd_handle(true);
    KokkosSparse::spadd_symbolic(
        &system_handle, solver.system_plus_constraints, transpose_matrix_full, solver.full_matrix
    );
    KokkosSparse::spadd_numeric(
        &system_handle, 1., solver.system_plus_constraints, 1., transpose_matrix_full,
        solver.full_matrix
    );

    Kokkos::parallel_for(
        "ConditionR", solver.num_system_dofs,
        ConditionR{
            solver.R,
            solver.conditioner,
        }
    );

    KokkosBlas::axpby(-1.0, solver.R, 0.0, solver.x);
    using GlobalCrsMatrixType = Tpetra::CrsMatrix<>;
    using size_type = GlobalCrsMatrixType::global_ordinal_type;
    using GlobalRowPtrType = GlobalCrsMatrixType::local_graph_device_type::row_map_type::non_const_type;
    using GlobalIndicesType = GlobalCrsMatrixType::local_graph_device_type::entries_type::non_const_type;
    using GlobalValuesType = GlobalCrsMatrixType::local_matrix_device_type::values_type;
    using GlobalMapType = GlobalCrsMatrixType::map_type;

    using GlobalMultiVectorType = Tpetra::MultiVector<>;
    using DualViewType = GlobalMultiVectorType::dual_view_type;

    auto comm = Tpetra::getDefaultComm ();

    auto A = std::invoke([&]{
      auto full_matrix = solver.full_matrix;
      auto rowMap = std::invoke([&](){
        const auto numLocalEntries = full_matrix.numRows();
        const auto numGlobalEntries = comm->getSize() * numLocalEntries;
        const auto indexBase = size_type{0u};
        return Teuchos::rcp(new GlobalMapType(numGlobalEntries, numLocalEntries, indexBase, comm));
      });

      auto colMap = std::invoke([&](){
        auto colInds = std::invoke([&](){
          const auto numLocalEntries = full_matrix.numRows();
          auto colInds_local = Kokkos::View<size_type*>("Column Map", numLocalEntries);
          auto colIndsMirror = Kokkos::create_mirror(colInds_local);
          for(int i = 0; i < numLocalEntries; ++i) {
            colIndsMirror(i) = rowMap->getGlobalElement(i);
          }
          Kokkos::deep_copy(colInds_local, colIndsMirror);
          return colInds_local;
        });
        const auto indexBase = size_type{0u};
        const auto INV = Teuchos::OrdinalTraits<size_type>::invalid ();
        return Teuchos::rcp(new GlobalMapType(INV, colInds, indexBase, comm));
      });
      
      auto rowPointers = GlobalRowPtrType("rowPointers", full_matrix.graph.row_map.extent(0));
      Kokkos::parallel_for("fillRowPtrs", rowPointers.extent(0), KOKKOS_LAMBDA(int i) {
          rowPointers(i) = full_matrix.graph.row_map(i);
      });
    
      auto columnIndices = GlobalIndicesType("columnIndices", full_matrix.graph.entries.extent(0));
      Kokkos::parallel_for("fillColumnIndices", columnIndices.extent(0), KOKKOS_LAMBDA(int i) {
          columnIndices(i) = full_matrix.graph.entries(i);
      });
      auto values = GlobalValuesType("values", full_matrix.values.extent(0));
      Kokkos::parallel_for("fillValues", values.extent(0), KOKKOS_LAMBDA(int i) {
          values(i) = full_matrix.values(i);
      });

      auto Ad = GlobalCrsMatrixType(rowMap, colMap, rowPointers, columnIndices, values);
      Ad.fillComplete();

      return Ad;
    });

    auto b = std::invoke([&](){
      auto b_lcl = DualViewType("b", solver.x.extent(0), 1);
      b_lcl.modify_device();
      Kokkos::deep_copy (Kokkos::subview (b_lcl.d_view, Kokkos::ALL (), 0), solver.x);
      return GlobalMultiVectorType(A.getRangeMap (), b_lcl);
    });

    auto x = std::invoke([&](){
      auto x_lcl = DualViewType("x", solver.x.extent(0), 1);
      x_lcl.modify_device();
      Kokkos::deep_copy (Kokkos::subview (x_lcl.d_view, Kokkos::ALL (), 0), solver.x);
      return GlobalMultiVectorType(A.getDomainMap (), x_lcl);
    });

    {
      auto sparse_region = Kokkos::Profiling::ScopedRegion("Sparse Solver");  
      auto amesos_solver = Amesos2::create<GlobalCrsMatrixType, GlobalMultiVectorType>("Basker", Teuchos::rcpFromRef(A), Teuchos::rcpFromRef(x), Teuchos::rcpFromRef(b));
      {
        auto symbolic_region = Kokkos::Profiling::ScopedRegion("Symbolic Factorization"); 
        amesos_solver->symbolicFactorization();
      }
      {
        auto numeric_region = Kokkos::Profiling::ScopedRegion("Numeric Factorization"); 
        amesos_solver->numericFactorization();
      }
      {
        auto solve_region = Kokkos::Profiling::ScopedRegion("Solve"); 
        amesos_solver->solve();
      }
    }
    Kokkos::deep_copy(solver.x, Kokkos::subview(x.getLocalViewDevice(Tpetra::Access::ReadOnly), Kokkos::ALL(), 0));

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
