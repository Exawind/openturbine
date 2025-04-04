#pragma once

#include "dss_handle_superlu.hpp"
#include "slu_ddefs.h"

namespace openturbine {
template <typename CrsMatrixType, typename MultiVectorType>
struct DSSSolveFunction<DSSHandle<DSSAlgorithm::SUPERLU>, CrsMatrixType, MultiVectorType> {
    static void solve(
        DSSHandle<DSSAlgorithm::SUPERLU>& dss_handle, CrsMatrixType&, MultiVectorType& b,
        MultiVectorType& x
    ) {
        auto& stat = dss_handle.get_stat();
        auto& L = dss_handle.get_L();
        auto& U = dss_handle.get_U();
        auto& perm_r = dss_handle.get_perm_r();
        auto& perm_c = dss_handle.get_perm_c();

        Kokkos::deep_copy(x, b);
        SuperMatrix Xmatrix;
        dCreate_Dense_Matrix(
            &Xmatrix, static_cast<int>(x.extent(0)), static_cast<int>(x.extent(1)), x.data(),
            static_cast<int>(x.extent(0)), SLU_DN, SLU_D, SLU_GE
        );
        auto info = 0;

        dgstrs(TRANS, &L, &U, perm_c.data(), perm_r.data(), &Xmatrix, &stat, &info);
    }
};

}  // namespace openturbine
