#pragma once

#include "OpenTurbine_config.h"

namespace openturbine {

enum class DSSAlgorithm : std::uint8_t {
    CUSOLVER_SP,
    CUDSS,
    KLU,
    UMFPACK,
    SUPERLU,
    MKL,
};

template <DSSAlgorithm>
class DSSHandle {
public:
    DSSHandle() = delete;
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
