#pragma once

#include <Amesos2.hpp>
#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "create_b_matrix.hpp"
#include "create_b_t_matrix.hpp"
#include "create_constraints_matrix_full.hpp"
#include "create_k_matrix.hpp"
#include "create_matrix_spadd.hpp"
#include "create_system_matrix_full.hpp"
#include "create_transpose_matrix_full.hpp"

namespace openturbine {

template <typename GlobalCrsMatrixType>
[[nodiscard]] inline Teuchos::RCP<GlobalCrsMatrixType> CreateFullMatrix(
    size_t num_system_dofs, size_t num_dofs, size_t num_constraint_dofs,
    const Kokkos::View<ConstraintType*>::const_type& constraint_type,
    const Kokkos::View<size_t* [6]>::const_type& base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& row_range,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& base_node_col_range,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& target_node_col_range,
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices
) {
    using CrsMatrixType = typename GlobalCrsMatrixType::local_matrix_device_type;
    using LocalOrdinalType = typename GlobalCrsMatrixType::local_ordinal_type;
    using GlobalOrdinalType = typename GlobalCrsMatrixType::global_ordinal_type;
    using ExecutionSpace = typename GlobalCrsMatrixType::execution_space;
    using MemorySpace = typename GlobalCrsMatrixType::memory_space;
    using KernelHandle = typename KokkosKernels::Experimental::KokkosKernelsHandle<
        typename CrsMatrixType::const_size_type, typename CrsMatrixType::const_ordinal_type,
        typename CrsMatrixType::const_value_type, ExecutionSpace, MemorySpace, MemorySpace>;
    //    KernelHandle spc_spadd_handle;
    //    KernelHandle full_system_spadd_handle;

    //    const auto constraints_matrix = CreateBMatrix<CrsMatrixType>(num_system_dofs,
    //    num_constraint_dofs, constraint_type, base_node_freedom_table, target_node_freedom_table,
    //    row_range, base_node_col_range, target_node_col_range); const auto B_t =
    //    CreateBtMatrix<CrsMatrixType>(num_system_dofs, num_constraint_dofs, constraint_type,
    //    base_node_freedom_table, target_node_freedom_table, row_range, base_node_col_range,
    //    target_node_col_range); const auto system_matrix =
    //    CreateKMatrix<CrsMatrixType>(num_system_dofs, node_freedom_allocation_table,
    //    node_freedom_map_table, num_nodes_per_element, node_state_indices); const auto
    //    system_matrix_full = CreateSystemMatrixFull(num_system_dofs, num_dofs, system_matrix);
    //    const auto constraints_matrix_full = CreateConstraintsMatrixFull(num_system_dofs, num_dofs,
    //    constraints_matrix); const auto transpose_matrix_full =
    //    CreateTransposeMatrixFull(num_system_dofs, num_dofs, B_t); const auto
    //    system_plus_constraints =
    // CreateMatrixSpadd(CreateSystemMatrixFull(num_system_dofs, num_dofs, system_matrix),
    //                                         CreateConstraintsMatrixFull(num_system_dofs, num_dofs,
    //                                         constraints_matrix), spc_spadd_handle);

    //    auto comm = Teuchos::createSerialComm<LocalOrdinalType>();
    //    auto row_map = ;
    //    auto col_map = Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(num_dofs, comm);

    // clang-format off
    return Teuchos::make_rcp<GlobalCrsMatrixType>(
               Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(
                   num_dofs, 
                   Teuchos::createSerialComm<LocalOrdinalType>()
               ),
               Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(
                   num_dofs,
                   Teuchos::createSerialComm<LocalOrdinalType>()
               ), 
               CreateMatrixSpadd<CrsMatrixType, KernelHandle>(
                   CreateMatrixSpadd<CrsMatrixType, KernelHandle>(
                       CreateSystemMatrixFull(
                           num_system_dofs,
                           num_dofs,
                           CreateKMatrix<CrsMatrixType>(
                               num_system_dofs,
                               node_freedom_allocation_table,
                               node_freedom_map_table,
                               num_nodes_per_element,
                               node_state_indices
                           )
                       ),
                       CreateConstraintsMatrixFull(
                           num_system_dofs,
                           num_dofs,
                           CreateBMatrix<CrsMatrixType>(
                               num_system_dofs,
                               num_constraint_dofs,
                               constraint_type,
                               base_node_freedom_table,
                               target_node_freedom_table,
                               row_range,
                               base_node_col_range,
                               target_node_col_range
                           )
                       )
                   ),
                   CreateTransposeMatrixFull(
                       num_system_dofs,
                       num_dofs,
                       CreateBtMatrix<CrsMatrixType>(
                           num_system_dofs,
                           num_constraint_dofs,
                           constraint_type,
                           base_node_freedom_table,
                           target_node_freedom_table,
                           row_range,
                           base_node_col_range,
                           target_node_col_range
                       )
                   )
               )
           );
    // clang-format on
}

}  // namespace openturbine
