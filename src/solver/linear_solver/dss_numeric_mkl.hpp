#pragma once

#include <mkl_dss.h>

namespace openturbine {

template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::MKL>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::MKL>& dss_handle, CrsMatrixType& A) {
        auto& handle = dss_handle.get_handle();
        constexpr MKL_INT opt = MKL_DSS_INDEFINITE;

        const auto* values = A.values.data();

        dss_factor_real(handle, opt, values);
    }
};

}  // namespace openturbine
