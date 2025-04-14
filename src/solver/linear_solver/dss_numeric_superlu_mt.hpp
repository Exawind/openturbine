#pragma once

#include "dss_handle_superlu_mt.hpp"
#include "slu_mt_ddefs.h"

namespace openturbine {
template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::SUPERLU_MT>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::SUPERLU_MT>& dss_handle, CrsMatrixType& A) {
        auto num_rows = A.numRows();
        auto num_cols = A.numCols();
        auto num_non_zero = A.nnz();

        auto* values = A.values.data();
        auto* row_ptrs = A.graph.row_map.data();
        auto* col_inds = A.graph.entries.data();

        auto& options = dss_handle.get_options();
        auto& stat = dss_handle.get_stat();
        auto& L = dss_handle.get_L();
        auto& U = dss_handle.get_U();

        SuperMatrix Amatrix;
        dCreate_CompCol_Matrix(
            &Amatrix, num_rows, num_cols, num_non_zero, values, col_inds, const_cast<int*>(row_ptrs),
            SLU_NC, SLU_D, SLU_GE
        );
        auto info = 0;

        SuperMatrix AC;
        sp_colorder(&Amatrix, options.perm_c, &options, &AC);

        pdgstrf(&options, &AC, options.perm_r, &L, &U, &stat, &info);

        Destroy_CompCol_Permuted(&AC);

        options.refact = YES;
    }
};

}  // namespace openturbine
