#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include <iostream>
// Include for ofstream
#include <fstream>

#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

inline void ElementalConstraintForcesResidual(
    LieGroupFieldView::const_type gen_coords, View1D constraints_residual
) {
    Kokkos::deep_copy(constraints_residual, 0.);
    // For the GEBT proof of concept problem (i.e. the clamped beam), the dofs are enforced to be
    // zero at the left end of the beam, so the constraint residual is simply based on the
    // generalized coordinates at the first node
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            // Construct rotation vector from root node rotation quaternion
            auto rotation_vector = openturbine::gen_alpha_solver::rotation_vector_from_quaternion(
                openturbine::gen_alpha_solver::Quaternion(
                    gen_coords(0, 3), gen_coords(0, 4), gen_coords(0, 5), gen_coords(0, 6)
                )
            );
            // Set residual as translation and rotation of root node
            // TODO: update when position & rotations are prescribed
            constraints_residual(0) = gen_coords(0, 0);
            constraints_residual(1) = gen_coords(0, 1);
            constraints_residual(2) = gen_coords(0, 2);
            constraints_residual(3) = rotation_vector.GetXComponent();
            constraints_residual(4) = rotation_vector.GetYComponent();
            constraints_residual(5) = rotation_vector.GetZComponent();
        }
    );
}

inline void AxialVector(
    View2D_3x3 matrix, View1D_Vector axial_vector
) {
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            axial_vector(0) = (matrix(2,1) - matrix(1,2)) / 2.;
            axial_vector(1) = (matrix(0,2) - matrix(2,0)) / 2.;
            axial_vector(2) = (matrix(1,0) - matrix(0,1)) / 2.;
        }
    );
}

inline void ConstraintResidualForRotatingBeam(
    LieGroupFieldView initial_position, LieGroupFieldView::const_type gen_coords,
    View1D applied_motion, View1D constraints_residual
) {
    // Constraint residual force for a rotating beam element with a prescribed rotation with
    // root/reference node positioned at hub radius of the rotor is defined as
    // {phi}_6x1 = {
    //     {phi_x}_3x1 = {u} + {X_0} - [R_BC] * {X_0},
    //     {phi_R}_3x1 = axial([R] * [R_BC]^T)
    // }
    // where,
    // {u} is the displacement vector of the constrained node,
    // {X_0} is the initial position vector of the constrained node,
    // [R_BC] is the relative rotation matrix of the reference node,
    // [R] is the relative rotation matrix of the constrained node

    // Relative rotation of reference node
    auto R_BC = openturbine::gen_alpha_solver::EulerParameterToRotationMatrix(
        Kokkos::subview(applied_motion, Kokkos::make_pair(3,7))
    );

    // Relative rotation of constrained node
    auto R = openturbine::gen_alpha_solver::EulerParameterToRotationMatrix(
        Kokkos::subview(gen_coords, 0, Kokkos::make_pair(3,7))
    );

    // {phi_x}_3x1 = {u} + {X_0} - [R_BC] * {X_0}
    auto u = Kokkos::subview(gen_coords, 0, Kokkos::make_pair(0,3));
    auto X0 = Kokkos::subview(initial_position, 0, Kokkos::make_pair(0,3));
    auto R_BC_X0 = View1D_Vector("R_BC_X0");
    KokkosBlas::gemv("N", 1., R_BC, X0, 0., R_BC_X0);

    // {phi_R}_3x1 = axial([R] * [R_BC]^T)
    auto R_R_BC_T = View2D_3x3("R_R_BC_T");
    KokkosBlas::gemm("N", "T", 1., R, R_BC, 0., R_R_BC_T);
    auto r_phi = View1D_Vector("r_phi");
    AxialVector(R_R_BC_T, r_phi);

    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            constraints_residual(0) = u(0) + X0(0) - R_BC_X0(0);
            constraints_residual(1) = u(1) + X0(1) - R_BC_X0(1);
            constraints_residual(2) = u(2) + X0(2) - R_BC_X0(2);
            constraints_residual(3) = r_phi(0);
            constraints_residual(4) = r_phi(1);
            constraints_residual(5) = r_phi(2);
        }
    );
}

}  // namespace openturbine::gebt_poc
