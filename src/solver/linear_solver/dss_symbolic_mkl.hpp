#pragma once

#include <mkl_dss.h>

#include "dss_handle_mkl.hpp"

namespace openturbine {
template <typename CrsMatrixType>
struct DSSSymbolicFunction<DSSHandle<DSSAlgorithm::MKL>, CrsMatrixType> {
    static void symbolic(DSSHandle<DSSAlgorithm::MKL>& dss_handle, CrsMatrixType& A) {
        auto& handle = dss_handle.get_handle();
        constexpr MKL_INT opt = MKL_DSS_NON_SYMMETRIC;

        const MKL_INT num_rows = A.numRows();
        const MKL_INT num_cols = A.numCols();
        const MKL_INT num_non_zeros = A.nnz();

        const auto* row_ptrs = A.graph.row_map.data();
        const auto* col_inds = A.graph.entries.data();

        dss_define_structure(handle, opt, row_ptrs, num_rows, num_cols, col_inds, num_non_zeros);

        auto& perm = dss_handle.get_perm();
        perm.resize(static_cast<size_t>(num_rows));
        constexpr MKL_INT reorder_opt = MKL_DSS_AUTO_ORDER;
        dss_reorder(handle, reorder_opt, perm.data());
    }
};
}  // namespace openturbine
