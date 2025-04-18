#pragma once

#include <mkl_dss.h>

namespace openturbine {

template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::MKL>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::MKL>& dss_handle, CrsMatrixType& A) {
        auto& handle = dss_handle.get_handle();
        constexpr MKL_INT opt = MKL_DSS_INDEFINITE;

        auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.values);

        dss_factor_real(handle, opt, values.data());
    }
};

}  // namespace openturbine
