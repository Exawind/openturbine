#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct UpdateLambdaPrediction {
    Kokkos::View<double*>::const_type lambda_delta;
    Kokkos::View<double*> lambda;

    KOKKOS_FUNCTION
    void operator()(const int i_lambda) const { lambda(i_lambda) += lambda_delta(i_lambda); }
};

}  // namespace openturbine
