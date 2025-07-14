#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename CrsMatrixType>
struct ContributeMassesToSparseMatrix {
    using DeviceType = typename CrsMatrixType::device_type;
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    using member_type =
        typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    double conditioner{};
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type element_freedom_signature;
    typename Kokkos::View<size_t* [6], DeviceType>::const_type element_freedom_table;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type dense;
    CrsMatrixType sparse;

    KOKKOS_FUNCTION
    void operator()(member_type member) const {
        const auto element = member.league_rank();
        constexpr auto is_sorted = true;
        constexpr auto force_atomic =
            !std::is_same_v<typename DeviceType::execution_space, Kokkos::Serial>;

        constexpr auto num_dofs = 6;
        constexpr auto hint = static_cast<typename CrsMatrixType::ordinal_type>(0);

        const auto first_column =
            static_cast<typename CrsMatrixType::ordinal_type>(element_freedom_table(element, 0));
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, num_dofs), [&](int component_1) {
            const auto row_num = static_cast<int>(element_freedom_table(element, component_1));
            auto row = sparse.row(row_num);
            auto offset = KokkosSparse::findRelOffset(
                &(row.colidx(0)), row.length, first_column, hint, is_sorted
            );
            for (auto component_2 = 0; component_2 < num_dofs; ++component_2, ++offset) {
                const auto contribution = dense(element, component_1, component_2) * conditioner;
                if constexpr (force_atomic) {
                    Kokkos::atomic_add(&(row.value(offset)), contribution);
                } else {
                    row.value(offset) += contribution;
                }
            }
        });
    }
};

}  // namespace openturbine
