#pragma once

#include "OpenTurbine_config.h"

namespace openturbine::dss {

template <typename DSHandleType, typename CrsMatrixType>
struct NumericFunction {
    NumericFunction() = delete;
};

}  // namespace openturbine::dss

#ifdef OpenTurbine_ENABLE_CUSOLVERSP
#include "dss_numeric_cusolversp.hpp"
#endif

#ifdef OpenTurbine_ENABLE_CUDSS
#include "dss_numeric_cudss.hpp"
#endif

#ifdef OpenTurbine_ENABLE_MKL
#include "dss_numeric_mkl.hpp"
#endif

#ifdef OpenTurbine_ENABLE_KLU
#include "dss_numeric_klu.hpp"
#endif

#ifdef OpenTurbine_ENABLE_UMFPACK
#include "dss_numeric_umfpack.hpp"
#endif

#ifdef OpenTurbine_ENABLE_SUPERLU
#include "dss_numeric_superlu.hpp"
#endif

#ifdef OpenTurbine_ENABLE_SUPERLU_MT
#include "dss_numeric_superlu_mt.hpp"
#endif

namespace openturbine::dss {

template <typename DSSHandleType, typename CrsMatrixType>
void numeric_factorization(DSSHandleType& dss_handle, CrsMatrixType& A) {
    NumericFunction<DSSHandleType, CrsMatrixType>::numeric(dss_handle, A);
}

}  // namespace openturbine::dss
