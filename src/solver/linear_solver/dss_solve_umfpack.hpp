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
        auto* values = A.values.data();
        auto* row_ptrs = A.graph.row_map.data();
        auto* col_inds = A.graph.entries.data();

        auto*& numeric = dss_handle.get_numeric();
        umfpack_di_solve(
            UMFPACK_At, row_ptrs, col_inds, values, x.data(), b.data(), numeric, nullptr, nullptr
        );
    }
};

}  // namespace openturbine
