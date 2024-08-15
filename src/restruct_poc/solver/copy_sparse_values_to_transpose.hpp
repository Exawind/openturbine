#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct CopySparseValuesToTranspose {
    using RowMapType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    CrsMatrixType input;
    RowMapType tmp;
    CrsMatrixType transpose;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto row_index = member.league_rank();
        const auto col_begin = input.graph.row_map(row_index);
        const auto col_end = input.graph.row_map(row_index + 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, col_begin, col_end), [&](int in_index) {
            const auto col_index = input.graph.entries(in_index);
            using atomic_incr_type = std::remove_reference_t<decltype(tmp(0))>;
            const auto pos = Kokkos::atomic_fetch_add(&(tmp(col_index)), atomic_incr_type{1});
            transpose.graph.entries(pos) = row_index;
            transpose.values(pos) = input.values(in_index);
        });
    }
};
}  // namespace openturbine
