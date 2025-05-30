#pragma once

#include "dss_handle_superlu_mt.hpp"
#include "slu_mt_ddefs.h"

namespace openturbine {
template <typename CrsMatrixType, typename MultiVectorType>
struct DSSSolveFunction<DSSHandle<DSSAlgorithm::SUPERLU_MT>, CrsMatrixType, MultiVectorType> {
    static void solve(
        DSSHandle<DSSAlgorithm::SUPERLU_MT>& dss_handle, CrsMatrixType&, MultiVectorType& b,
        MultiVectorType& x
    ) {
        auto& options = dss_handle.get_options();
        auto& stat = dss_handle.get_stat();
        auto& L = dss_handle.get_L();
        auto& U = dss_handle.get_U();

        Kokkos::deep_copy(x, b);

        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);

        SuperMatrix Xmatrix;
        dCreate_Dense_Matrix(
            &Xmatrix, static_cast<int>(x.extent(0)), static_cast<int>(x.extent(1)), x_host.data(),
            static_cast<int>(x.extent(0)), SLU_DN, SLU_D, SLU_GE
        );

        auto info = 0;
        dgstrs(TRANS, &L, &U, options.perm_r, options.perm_c, &Xmatrix, &stat, &info);

        Kokkos::deep_copy(x, x_host);
    }
};

}  // namespace openturbine
