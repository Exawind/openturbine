#pragma once

#include <umfpack.h>

#include "dss_handle_umfpack.hpp"

namespace openturbine {
template <typename CrsMatrixType, typename MultiVectorType>
struct DSSSolveFunction<DSSHandle<DSSAlgorithm::UMFPACK>, CrsMatrixType, MultiVectorType> {
    static void solve(
        DSSHandle<DSSAlgorithm::UMFPACK>& dss_handle, CrsMatrixType& A, MultiVectorType& b,
        MultiVectorType& x
    ) {
        auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.values);
        auto row_ptrs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.row_map);
        auto col_inds = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.entries);

        auto x_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), x);
        auto b_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

        auto*& numeric = dss_handle.get_numeric();
        umfpack_di_solve(
            UMFPACK_At, row_ptrs.data(), col_inds.data(), values.data(), x_host.data(),
            b_host.data(), numeric, nullptr, nullptr
        );

        Kokkos::deep_copy(x, x_host);
    }
};

}  // namespace openturbine
