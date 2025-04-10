#pragma once

#include "dss_handle.hpp"

namespace openturbine {

template <typename DSSHandleType, typename CrsMatrixType>
struct DSSSymbolicFunction {
    DSSSymbolicFunction() = delete;
};

}  // namespace openturbine

#ifdef OpenTurbine_ENABLE_CUSOLVERSP
#include "dss_symbolic_cusolversp.hpp"
#endif

#ifdef OpenTurbine_ENABLE_CUDSS
#include "dss_symbolic_cudss.hpp"
#endif

#ifdef OpenTurbine_ENABLE_MKL
#include "dss_symbolic_mkl.hpp"
#endif

#ifdef OpenTurbine_ENABLE_KLU
#include "dss_symbolic_klu.hpp"
#endif

#ifdef OpenTurbine_ENABLE_UMFPACK
#include "dss_symbolic_umfpack.hpp"
#endif

#ifdef OpenTurbine_ENABLE_SUPERLU
#include "dss_symbolic_superlu.hpp"
#endif

#ifdef OpenTurbine_ENABLE_SUPERLU_MT
#include "dss_symbolic_superlu_mt.hpp"
#endif

namespace openturbine {

template <typename DSSHandleType, typename CrsMatrixType>
void dss_symbolic(DSSHandleType& dss_handle, CrsMatrixType& A) {
    DSSSymbolicFunction<DSSHandleType, CrsMatrixType>::symbolic(dss_handle, A);
}

}  // namespace openturbine
