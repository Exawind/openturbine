#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

struct CalculateErrorSumSquares {
    using value_type = double;

    double atol;
    double rtol;
    View_Nx6::const_type q_delta;
    View_N::const_type x;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& err) const {
        err += Kokkos::pow(x(i) / (atol + rtol * Kokkos::abs(q_delta(i / 6, i % 6))), 2.);
    }
};

}
