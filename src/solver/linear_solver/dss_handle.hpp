#pragma once

#include "dss_algorithm.hpp"

#include "OpenTurbine_config.h"

namespace openturbine::dss {

template <Algorithm>
class Handle {
public:
    Handle() = delete;
};
}  // namespace openturbine

#ifdef OpenTurbine_ENABLE_CUSOLVERSP
#include "dss_handle_cusolversp.hpp"
#endif

#ifdef OpenTurbine_ENABLE_CUDSS
#include "dss_handle_cudss.hpp"
#endif

#ifdef OpenTurbine_ENABLE_MKL
#include "dss_handle_mkl.hpp"
#endif

#ifdef OpenTurbine_ENABLE_KLU
#include "dss_handle_klu.hpp"
#endif

#ifdef OpenTurbine_ENABLE_UMFPACK
#include "dss_handle_umfpack.hpp"
#endif

#ifdef OpenTurbine_ENABLE_SUPERLU
#include "dss_handle_superlu.hpp"
#endif

#ifdef OpenTurbine_ENABLE_SUPERLU_MT
#include "dss_handle_superlu_mt.hpp"
#endif
