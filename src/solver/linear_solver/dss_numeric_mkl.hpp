#pragma once

#include <mkl.h>
#include <mkl_pardiso.h>

namespace openturbine {

template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::MKL>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::MKL>& dss_handle, CrsMatrixType& A) {
        const MKL_INT symbolic_phase = 22;

        const MKL_INT num_rows = A.numRows();

        auto* values = A.values.data();
        auto* row_ptrs = A.graph.row_map.data();
        auto* col_inds = A.graph.entries.data();

        auto** handle = dss_handle.get_handle();
        auto* iparm = dss_handle.get_iparm();
        auto& mtype = dss_handle.get_mtype();
        auto& msglvl = dss_handle.get_msglvl();
        auto& maxfct = dss_handle.get_maxfct();
        auto& nrhs = dss_handle.get_nrhs();
        auto& mnum = dss_handle.get_mnum();
        auto& perm = dss_handle.get_perm();
        MKL_INT error;

        pardiso(
            handle, &maxfct, &mnum, &mtype, &symbolic_phase, &num_rows, values, row_ptrs, col_inds,
            perm.data(), &nrhs, iparm, &msglvl, nullptr, nullptr, &error
        );
    }
};

}  // namespace openturbine
