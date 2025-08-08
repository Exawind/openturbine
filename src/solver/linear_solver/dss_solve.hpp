#pragma once

#include "OpenTurbine_config.h"

namespace openturbine::dss {

template <typename DSSHandleType, typename CrsMatrixType, typename MultiVectorType>
struct SolveFunction {
    SolveFunction() = delete;
};

}  // namespace openturbine::dss

#ifdef OpenTurbine_ENABLE_CUSOLVERSP
#include "dss_solve_cusolversp.hpp"
#endif

#ifdef OpenTurbine_ENABLE_CUDSS
#include "dss_solve_cudss.hpp"
#endif

#ifdef OpenTurbine_ENABLE_MKL
#include "dss_solve_mkl.hpp"
#endif

#ifdef OpenTurbine_ENABLE_KLU
#include "dss_solve_klu.hpp"
#endif

#ifdef OpenTurbine_ENABLE_UMFPACK
#include "dss_solve_umfpack.hpp"
#endif

#ifdef OpenTurbine_ENABLE_SUPERLU
#include "dss_solve_superlu.hpp"
#endif

#ifdef OpenTurbine_ENABLE_SUPERLU_MT
#include "dss_solve_superlu_mt.hpp"
#endif

namespace openturbine::dss {

template <typename DSSHandleType, typename CrsMatrixType, typename MultiVectorType>
void solve(DSSHandleType& dss_handle, CrsMatrixType& A, MultiVectorType& b, MultiVectorType& x) {
    SolveFunction<DSSHandleType, CrsMatrixType, MultiVectorType>::solve(dss_handle, A, b, x);
}

}  // namespace openturbine::dss
