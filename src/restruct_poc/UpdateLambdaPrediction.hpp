#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

struct UpdateLambdaPrediction {
    View_N::const_type lambda_delta;
    View_N lambda;

    KOKKOS_FUNCTION
    void operator()(const int i_lambda) const { lambda(i_lambda) -= lambda_delta(i_lambda); }
};

}
