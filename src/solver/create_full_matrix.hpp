#pragma once

#include <Amesos2.hpp>
#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "create_constraints_matrix_full.hpp"
#include "create_matrix_spadd.hpp"
#include "create_system_matrix_full.hpp"
#include "create_transpose_matrix_full.hpp"

namespace openturbine {

template <typename GlobalCrsMatrixType>
[[nodiscard]] inline Teuchos::RCP<GlobalCrsMatrixType> CreateFullMatrix(
    size_t num_system_dofs, size_t num_dofs,
    const Kokkos::View<size_t*>::const_type& base_active_dofs,
    const Kokkos::View<size_t*>::const_type& target_active_dofs,
    const Kokkos::View<size_t* [6]>::const_type& base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& row_range,
    const Kokkos::View<size_t*>::const_type& active_dofs,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full Matrix");

    using CrsMatrixType = typename GlobalCrsMatrixType::local_matrix_device_type;
    using LocalOrdinalType = typename GlobalCrsMatrixType::local_ordinal_type;
    using GlobalOrdinalType = typename GlobalCrsMatrixType::global_ordinal_type;
    using ExecutionSpace = typename GlobalCrsMatrixType::execution_space;
    using MemorySpace = typename GlobalCrsMatrixType::memory_space;
    using KernelHandle = typename KokkosKernels::Experimental::KokkosKernelsHandle<
        typename CrsMatrixType::const_size_type, typename CrsMatrixType::const_ordinal_type,
        typename CrsMatrixType::const_value_type, ExecutionSpace, MemorySpace, MemorySpace>;

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
                       CreateTransposeMatrixFull<CrsMatrixType>(
                           num_system_dofs,
                           num_dofs,
                           active_dofs,
                           node_freedom_map_table,
                           base_active_dofs,
                           target_active_dofs,
                           base_node_freedom_table,
                           target_node_freedom_table,
                           row_range
                       ),
                       CreateConstraintsMatrixFull<CrsMatrixType>(
                           num_system_dofs,
                           num_dofs,
                           base_active_dofs,
                           target_active_dofs,
                           base_node_freedom_table,
                           target_node_freedom_table,
                           row_range
                       )
                   ),
                   CreateSystemMatrixFull<CrsMatrixType>(
                       num_system_dofs,
                       num_dofs,
                       active_dofs,
                       node_freedom_map_table,
                       num_nodes_per_element,
                       node_state_indices
                   )
               )
           );
    // clang-format on
}

}  // namespace openturbine
