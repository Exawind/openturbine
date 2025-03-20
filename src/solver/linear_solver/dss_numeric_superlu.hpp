#pragma once

#include "dss_handle_superlu.hpp"
#include "slu_ddefs.h"

namespace openturbine {
template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::SUPERLU>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::SUPERLU>& dss_handle, CrsMatrixType& A) {
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
        auto& Glu = dss_handle.get_Glu();
        auto& perm_r = dss_handle.get_perm_r();
        auto& perm_c = dss_handle.get_perm_c();
        auto& etree = dss_handle.get_etree();
        auto& work = dss_handle.get_work();

        SuperMatrix Amatrix;
        dCreate_CompCol_Matrix(
            &Amatrix, num_rows, num_cols, num_non_zero, values, col_inds, const_cast<int*>(row_ptrs),
            SLU_NC, SLU_D, SLU_GE
        );
        constexpr auto relax = 1;
        constexpr auto panel_size = 1;
        auto info = 0;

        SuperMatrix AC;
        sp_preorder(&options, &Amatrix, perm_c.data(), etree.data(), &AC);

        dgstrf(
            &options, &AC, relax, panel_size, etree.data(), work.data(),
            static_cast<int>(work.size()), perm_c.data(), perm_r.data(), &L, &U, &Glu, &stat, &info
        );

        Destroy_CompCol_Permuted(&AC);

        options.Fact = SamePattern;
    }
};

}  // namespace openturbine
