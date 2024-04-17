#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

struct UpdateDynamicPrediction {
    double h;
    double beta_prime;
    double gamma_prime;
    View_N::const_type x_delta;
    View_Nx6 q_delta;
    View_Nx6 v;
    View_Nx6 vd;

    KOKKOS_FUNCTION
    void operator()(const int i_node) const {
        for (int j = 0; j < kLieAlgebraComponents; j++) {
            double delta = x_delta(i_node * 6 + j);
            q_delta(i_node, j) += delta / h;
            v(i_node, j) += gamma_prime * delta;
            vd(i_node, j) += beta_prime * delta;
        }
    }
};

}
