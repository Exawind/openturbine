#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct UpdateStaticPrediction {
    double h;
    double beta_prime;
    double gamma_prime;
    Kokkos::View<double*>::const_type x_delta;
    Kokkos::View<double* [6]> q_delta;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        for (auto j = 0U; j < 6U; j++) {
            const auto delta = x_delta(i_node * 6 + j);
            q_delta(i_node, j) += delta / h;
        }
    }
};

}  // namespace openturbine
