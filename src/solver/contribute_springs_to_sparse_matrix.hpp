#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {
template <typename CrsMatrixType>
struct ContributeSpringsToSparseMatrix {
    using DeviceType = typename CrsMatrixType::device_type;
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    using member_type =
        typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    double conditioner{};
    typename Kokkos::View<FreedomSignature* [2], DeviceType>::const_type element_freedom_signature;
    typename Kokkos::View<size_t* [2][3], DeviceType>::const_type element_freedom_table;
    typename Kokkos::View<double* [2][2][3][3], DeviceType>::const_type
        dense;             //< Element Stiffness matrices
    CrsMatrixType sparse;  //< Global sparse stiffness matrix

    KOKKOS_FUNCTION
    void operator()(member_type member) const {
        const auto element = member.league_rank();
        constexpr auto is_sorted = true;
        constexpr auto force_atomic =
            !std::is_same_v<typename DeviceType::execution_space, Kokkos::Serial>;

        constexpr auto num_dofs = 3;
        constexpr auto num_nodes = 2;
        constexpr auto hint = static_cast<typename CrsMatrixType::ordinal_type>(0);

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, num_nodes * num_nodes),
            [&](int node_12) {
                const auto node_1 = node_12 / num_nodes;
                const auto node_2 = node_12 % num_nodes;
                const auto first_column = static_cast<typename CrsMatrixType::ordinal_type>(
                    element_freedom_table(element, node_2, 0)
                );

                for (auto component_1 = 0; component_1 < num_dofs; ++component_1) {
                    const auto row_num =
                        static_cast<int>(element_freedom_table(element, node_1, component_1));
                    auto row = sparse.row(row_num);
                    auto offset = KokkosSparse::findRelOffset(
                        &(row.colidx(0)), row.length, first_column, hint, is_sorted
                    );
                    for (auto component_2 = 0; component_2 < num_dofs; ++component_2, ++offset) {
                        const auto contribution =
                            dense(element, node_1, node_2, component_1, component_2) * conditioner;
                        if constexpr (force_atomic) {
                            Kokkos::atomic_add(&(row.value(offset)), contribution);
                        } else {
                            row.value(offset) += contribution;
                        }
                    }
                }
            }
        );
    };
};

}  // namespace openturbine
