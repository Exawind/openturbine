#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct UpdateStaticPrediction {
    double h;
    double beta_prime;
    double gamma_prime;
    View_N::const_type x_delta;
    View_Nx6 q_delta;

    KOKKOS_FUNCTION
    void operator()(const int i_node) const {
        for (int j = 0; j < kLieAlgebraComponents; j++) {
            double delta = x_delta(i_node * 6 + j);
            q_delta(i_node, j) += delta / h;
        }
    }
};

}  // namespace openturbine
