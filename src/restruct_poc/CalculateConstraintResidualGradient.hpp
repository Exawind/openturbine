#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<Constraints::NodeIndices*>::const_type node_indices;
    View_Nx3::const_type constraint_X0;
    View_Nx7::const_type constraint_u;
    View_Nx7::const_type node_u;
    View_N Phi;
    View_NxN B;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto i_node1 = node_indices(i_constraint).base_node_index;
        auto i_node2 = node_indices(i_constraint).constrained_node_index;

        Quaternion R1;
        if (i_node1 == -1) {
            R1 = Quaternion(
                constraint_u(i_constraint, 3), constraint_u(i_constraint, 4),
                constraint_u(i_constraint, 5), constraint_u(i_constraint, 6)
            );
        } else {
            R1 = Quaternion(
                node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)
            );
        }

        Quaternion R2(
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)
        );

        Vector X0(
            constraint_X0(i_constraint, 0), constraint_X0(i_constraint, 1),
            constraint_X0(i_constraint, 2)
        );

        Vector u2(node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2));

        auto i_row = i_constraint * kLieAlgebraComponents;
        auto i_col = i_node2 * kLieAlgebraComponents;

        auto Phi_x = u2 + X0 - R1 * X0;
        auto Phi_p = (R2 * R1.GetInverse()).Axial();

        Phi(i_row + 0) = Phi_x.GetXComponent();
        Phi(i_row + 1) = Phi_x.GetYComponent();
        Phi(i_row + 2) = Phi_x.GetZComponent();

        Phi(i_row + 3) = Phi_p.GetXComponent();
        Phi(i_row + 4) = Phi_p.GetYComponent();
        Phi(i_row + 5) = Phi_p.GetZComponent();

        auto R2R1T = (R1 * R2.GetInverse());
        auto mR2R1T = R2R1T.to_rotation_matrix();
        auto tr = R2R1T.Trace();
        RotationMatrix trI3(tr, 0., 0., 0., tr, 0., 0., 0., tr);

        for (int i = 0; i < 3; ++i) {
            B(i_row + i, i_col + i) = 1.;
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B(i_row + i + 3, i_col + j + 3) = (trI3(i, j) - mR2R1T(i, j)) / 2.;
            }
        }
    }
};

}
