#pragma once

#include <klu.h>

#include "dss_handle_klu.hpp"

namespace openturbine {
template <typename CrsMatrixType, typename MultiVectorType>
struct DSSSolveFunction<DSSHandle<DSSAlgorithm::KLU>, CrsMatrixType, MultiVectorType> {
    static void solve(
        DSSHandle<DSSAlgorithm::KLU>& dss_handle, CrsMatrixType& A, MultiVectorType& b,
        MultiVectorType& x
    ) {
        const auto num_rows = A.numRows();

        auto*& symbolic = dss_handle.get_symbolic();
        auto*& numeric = dss_handle.get_numeric();
        auto& common = dss_handle.get_common();

        Kokkos::deep_copy(x, b);
        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);

        klu_tsolve(symbolic, numeric, num_rows, 1, x_host.data(), &common);
        Kokkos::deep_copy(x, x_host);
    }
};

}  // namespace openturbine
