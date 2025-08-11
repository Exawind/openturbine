#pragma once
#include <cstdint>

namespace openturbine::dss {

enum class Algorithm : std::uint8_t {
    CUSOLVER_SP,
    CUDSS,
    KLU,
    UMFPACK,
    SUPERLU,
    SUPERLU_MT,
    MKL,
    NONE,
};

}  // namespace openturbine::dss
