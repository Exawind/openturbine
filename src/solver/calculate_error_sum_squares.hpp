#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct CalculateErrorSumSquares {
    using value_type = double;

    double atol;
    double rtol;
    Kokkos::View<double* [6]>::const_type q_delta;
    Kokkos::View<double*>::const_type x;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& err) const {
        err += Kokkos::pow(x(i) / (atol + rtol * Kokkos::abs(q_delta(i / 6, i % 6))), 2.);
    }
};

}  // namespace openturbine
