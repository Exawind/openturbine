#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateDisplacement {
    double h;
    View_Nx6 q_delta;
    View_Nx7 q_prev;
    View_Nx7 q;

    KOKKOS_FUNCTION
    void operator()(const int i_node) const {
        q(i_node, 0) = q_prev(i_node, 0) + h * q_delta(i_node, 0);
        q(i_node, 1) = q_prev(i_node, 1) + h * q_delta(i_node, 1);
        q(i_node, 2) = q_prev(i_node, 2) + h * q_delta(i_node, 2);

        auto quat_delta = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(
            h * q_delta(i_node, 3), h * q_delta(i_node, 4), h * q_delta(i_node, 5)
        );

        Quaternion quat_prev(
            q_prev(i_node, 3), q_prev(i_node, 4), q_prev(i_node, 5), q_prev(i_node, 6)
        );

        auto quat_new = quat_delta * quat_prev;

        q(i_node, 3) = quat_new.GetScalarComponent();
        q(i_node, 4) = quat_new.GetXComponent();
        q(i_node, 5) = quat_new.GetYComponent();
        q(i_node, 6) = quat_new.GetZComponent();
    }
};

}  // namespace openturbine
