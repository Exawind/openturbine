#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename CrsMatrixType>
struct ContributeBeamsToSparseMatrix {
    using DeviceType = typename CrsMatrixType::device_type;
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    using member_type =
        typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    double conditioner{};
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<FreedomSignature**, DeviceType>::const_type element_freedom_signature;
    typename Kokkos::View<size_t** [6], DeviceType>::const_type element_freedom_table;
    typename Kokkos::View<double*** [6][6], DeviceType>::const_type dense;
    CrsMatrixType sparse;

    KOKKOS_FUNCTION
    void operator()(member_type member) const {
        const auto i = member.league_rank();
        const auto num_nodes = static_cast<int>(num_nodes_per_element(i));
        constexpr auto is_sorted = true;
        constexpr auto force_atomic =
            !std::is_same_v<typename DeviceType::execution_space, Kokkos::Serial>;
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, num_nodes * num_nodes),
            [&](int node_12) {
                const auto node_1 = node_12 % num_nodes;
                const auto node_2 = node_12 / num_nodes;
                constexpr auto num_dofs = 6;
                const auto hint =
                    static_cast<typename CrsMatrixType::ordinal_type>(node_2 * num_dofs);

                const auto first_column = static_cast<typename CrsMatrixType::ordinal_type>(
                    element_freedom_table(i, node_2, 0)
                );
                const auto local_dense =
                    Kokkos::subview(dense, i, node_1, node_2, Kokkos::ALL, Kokkos::ALL);
                for (auto component_1 = 0; component_1 < num_dofs; ++component_1) {
                    const auto row_num =
                        static_cast<int>(element_freedom_table(i, node_1, component_1));
                    auto row = sparse.row(row_num);
                    auto offset = KokkosSparse::findRelOffset(
                        &(row.colidx(0)), row.length, first_column, hint, is_sorted
                    );
                    auto* matrix_ptr = &(row.value(offset));
                    for (auto component_2 = 0; component_2 < num_dofs; ++component_2, ++matrix_ptr) {
                        const auto contribution =
                            local_dense(component_1, component_2) * conditioner;
                        if constexpr (force_atomic) {
                            Kokkos::atomic_add(matrix_ptr, contribution);
                        } else {
                            *matrix_ptr += contribution;
                        }
                    }
                }
            }
        );
    }
};

}  // namespace openturbine
