#pragma once
#include <cstdint>

namespace openturbine {

enum class DSSAlgorithm : std::uint8_t {
    CUSOLVER_SP,
    CUDSS,
    KLU,
    UMFPACK,
    SUPERLU,
    SUPERLU_MT,
    MKL,
    NONE,
};

}  // namespace openturbine
