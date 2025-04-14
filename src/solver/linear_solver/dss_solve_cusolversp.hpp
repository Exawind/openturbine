#pragma once

#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include "dss_handle_cusolversp.hpp"

namespace openturbine {
template <typename CrsMatrixType, typename MultiVectorType>
struct DSSSolveFunction<DSSHandle<DSSAlgorithm::CUSOLVER_SP>, CrsMatrixType, MultiVectorType> {
    static void solve(
        DSSHandle<DSSAlgorithm::CUSOLVER_SP>& dss_handle, CrsMatrixType& A, MultiVectorType& b,
        MultiVectorType& x
    ) {
        const auto num_rows = A.numRows();
        const auto num_cols = A.numCols();
        const auto num_non_zero = static_cast<int>(A.nnz());

        auto* values = A.values.data();
        auto* row_ptrs = A.graph.row_map.data();
        auto* col_inds = A.graph.entries.data();

        auto* b_values = b.data();
        auto* x_values = x.data();

        auto& handle = dss_handle.get_handle();
        auto& info = dss_handle.get_info();
        auto& buffer = dss_handle.get_buffer();

        cusolverSpDcsrqrSolve(handle, num_rows, num_cols, b_values, x_values, info, buffer.data());
    }
};

}  // namespace openturbine
