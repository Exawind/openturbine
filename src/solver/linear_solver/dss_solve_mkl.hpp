#pragma once

#include <mkl_dss.h>

#include "dss_handle_mkl.hpp"

namespace openturbine {
template <typename CrsMatrixType, typename MultiVectorType>
struct DSSSolveFunction<DSSHandle<DSSAlgorithm::MKL>, CrsMatrixType, MultiVectorType> {
    static void solve(
        DSSHandle<DSSAlgorithm::MKL>& dss_handle, CrsMatrixType&, MultiVectorType& b,
        MultiVectorType& x
    ) {
        auto& handle = dss_handle.get_handle();
        constexpr MKL_INT opt = 0;

        const auto nrhs = static_cast<MKL_INT>(b.extent(1));

        dss_solve_real(handle, opt, b.data(), nrhs, x.data());
    }
};
}  // namespace openturbine
