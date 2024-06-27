#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
struct CopyIntoSparseMatrix {
    using crs_matrix_type = KokkosSparse::CrsMatrix<
        double, int,
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>>;
    using row_data_type = Kokkos::View<
        double*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using col_idx_type = Kokkos::View<
        int*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    crs_matrix_type sparse;
    Kokkos::View<const double**> dense;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        auto i = member.league_rank();
        auto row = sparse.row(i);
        auto row_map = sparse.graph.row_map;
        auto cols = sparse.graph.entries;
        auto row_data = row_data_type(member.team_scratch(1), row.length);
        auto col_idx = col_idx_type(member.team_scratch(1), row.length);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
            col_idx(entry) = cols(row_map(i) + entry);
            row_data(entry) = dense(i, col_idx(entry));
        });
        member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
            sparse.replaceValues(i, col_idx.data(), row.length, row_data.data());
        });
    }
};
}  // namespace openturbine