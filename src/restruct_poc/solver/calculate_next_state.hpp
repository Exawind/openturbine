#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateNextState {
    double h;
    double alpha_f;
    double alpha_m;
    double beta;
    double gamma;
    View_Nx6 q_delta;
    View_Nx6 v;
    View_Nx6 vd;
    View_Nx6 a;

    KOKKOS_FUNCTION
    void operator()(const size_t i) const {
        for (auto j = 0U; j < kLieAlgebraComponents; ++j) {
            const double v_p = v(i, j);    // Save velocity from previous iteration
            const double vd_p = vd(i, j);  // Save acceleration from previous iteration
            const double a_p = a(i, j);    // Save algorithmic acceleration from previous iteration
            vd(i, j) = 0.;
            a(i, j) = (alpha_f * vd_p - alpha_m * a_p) / (1. - alpha_m);
            v(i, j) = v_p + h * (1. - gamma) * a_p + gamma * h * a(i, j);
            q_delta(i, j) = v_p + (0.5 - beta) * h * a_p + beta * h * a(i, j);
        }
    }
};

}  // namespace openturbine
