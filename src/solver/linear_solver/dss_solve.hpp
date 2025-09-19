#pragma once

#include "Kynema_config.h"

namespace kynema::dss {

template <typename DSSHandleType, typename CrsMatrixType, typename MultiVectorType>
struct SolveFunction {
    SolveFunction() = delete;
};

}  // namespace kynema::dss

#ifdef Kynema_ENABLE_CUSOLVERSP
#include "dss_solve_cusolversp.hpp"
#endif

#ifdef Kynema_ENABLE_CUDSS
#include "dss_solve_cudss.hpp"
#endif

#ifdef Kynema_ENABLE_MKL
#include "dss_solve_mkl.hpp"
#endif

#ifdef Kynema_ENABLE_KLU
#include "dss_solve_klu.hpp"
#endif

#ifdef Kynema_ENABLE_UMFPACK
#include "dss_solve_umfpack.hpp"
#endif

#ifdef Kynema_ENABLE_SUPERLU
#include "dss_solve_superlu.hpp"
#endif

#ifdef Kynema_ENABLE_SUPERLU_MT
#include "dss_solve_superlu_mt.hpp"
#endif

namespace kynema::dss {

template <typename DSSHandleType, typename CrsMatrixType, typename MultiVectorType>
void solve(DSSHandleType& dss_handle, CrsMatrixType& A, MultiVectorType& b, MultiVectorType& x) {
    SolveFunction<DSSHandleType, CrsMatrixType, MultiVectorType>::solve(dss_handle, A, b, x);
}

}  // namespace kynema::dss
