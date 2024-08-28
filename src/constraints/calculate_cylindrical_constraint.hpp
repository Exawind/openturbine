#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/math/quaternion_operations.hpp"
#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateCylindricalConstraint {
    Kokkos::View<size_t* [2]>::const_type node_index;
    Kokkos::View<size_t* [2]>::const_type row_range;
    Kokkos::View<size_t* [2][2]>::const_type node_col_range;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [3][3]>::const_type axis;
    Kokkos::View<double*>::const_type control;
    Kokkos::View<double* [7]>::const_type constraint_u;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double*> Phi_;
    Kokkos::View<double* [6][12]> gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        const auto i_node1 = node_index(i_constraint, 0);
        const auto i_node2 = node_index(i_constraint, 1);

        // Initial difference between nodes
        const auto X0_data = Kokkos::Array<double, 3>{
            X0_(i_constraint, 0), X0_(i_constraint, 1), X0_(i_constraint, 2)};
        const auto X0 = View_3::const_type{X0_data.data()};

        // Base node displacement
        const auto u1_data =
            Kokkos::Array<double, 3>{node_u(i_node1, 0), node_u(i_node1, 1), node_u(i_node1, 2)};
        const auto u1 = View_3::const_type{u1_data.data()};
        const auto R1_data = Kokkos::Array<double, 4>{
            node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)};
        const auto R1 = Kokkos::View<double[4]>::const_type{R1_data.data()};

        // Target node displacement
        const auto R2_data = Kokkos::Array<double, 4>{
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)};
        const auto R2 = Kokkos::View<double[4]>::const_type{R2_data.data()};
        const auto u2_data =
            Kokkos::Array<double, 3>{node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2)};
        const auto u2 = View_3::const_type{u2_data.data()};

        // Rotation control
        auto R1t_data = Kokkos::Array<double, 4>{};
        auto R1t = Kokkos::View<double[4]>{R1t_data.data()};

        auto R1_X0_data = Kokkos::Array<double, 4>{};
        auto R1_X0 = Kokkos::View<double[4]>{R1_X0_data.data()};

        // Cylindrical constraint data
        const auto x0_data = Kokkos::Array<double, 3>{
            axis(i_constraint, 0, 0), axis(i_constraint, 0, 1), axis(i_constraint, 0, 2)};
        const auto x0 = View_3::const_type{x0_data.data()};
        const auto y0_data = Kokkos::Array<double, 3>{
            axis(i_constraint, 1, 0), axis(i_constraint, 1, 1), axis(i_constraint, 1, 2)};
        const auto y0 = View_3::const_type{y0_data.data()};
        const auto z0_data = Kokkos::Array<double, 3>{
            axis(i_constraint, 2, 0), axis(i_constraint, 2, 1), axis(i_constraint, 2, 2)};
        const auto z0 = View_3::const_type{z0_data.data()};
        auto x_data = Kokkos::Array<double, 3>{};
        const auto x = View_3{x_data.data()};
        auto y_data = Kokkos::Array<double, 3>{};
        const auto y = View_3{y_data.data()};
        auto z_data = Kokkos::Array<double, 3>{};
        const auto z = View_3{z_data.data()};
        auto xcy_data = Kokkos::Array<double, 3>{};
        const auto xcy = View_3{xcy_data.data()};
        auto xcz_data = Kokkos::Array<double, 3>{};
        const auto xcz = View_3{xcz_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        // Extract residual rows relevant to this constraint
        const auto Phi = Kokkos::subview(
            Phi_, Kokkos::make_pair(row_range(i_constraint, 0), row_range(i_constraint, 1))
        );

        // Phi(0:3) = u2 + X0 - u1 - R1*X0
        QuaternionInverse(R1, R1t);
        RotateVectorByQuaternion(R1, X0, R1_X0);
        for (int i = 0; i < 3; ++i) {
            Phi(i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
        }

        // Angular residual
        RotateVectorByQuaternion(R1, x0, x);
        RotateVectorByQuaternion(R2, y0, y);
        RotateVectorByQuaternion(R2, z0, z);
        // Phi(3) = dot(R2 * z0_hat, R1 * x0_hat)
        Phi(3) = DotProduct(z, x);
        // Phi(4) = dot(R2 * y0_hat, R1 * x0_hat)
        Phi(4) = DotProduct(y, x);

        //----------------------------------------------------------------------
        // Constraint Gradient Matrix
        //----------------------------------------------------------------------

        //---------------------------------
        // Target Node
        //---------------------------------
        {
            // Extract gradient block for target node of this constraint
            const auto B = Kokkos::subview(
                gradient_terms, i_constraint, Kokkos::ALL,
                Kokkos::make_pair(
                    node_col_range(i_constraint, 1, 0), node_col_range(i_constraint, 1, 1)
                )
            );

            // B(0:3,0:3) = I
            for (int i = 0; i < 3; ++i) {
                B(i, i) = 1.;
            }

            // B(3,3:6) = -cross(R1 * x0_hat, transpose(R2 * z0_hat))
            CrossProduct(x, z, xcz);
            // B(4,3:6) = -cross(R1 * x0_hat, transpose(R2 * y0_hat))
            CrossProduct(x, y, xcy);
            for (int j = 0; j < 3; ++j) {
                B(3, j + 3) = -xcz(j);
                B(4, j + 3) = -xcy(j);
            }
        }
        //---------------------------------
        // Base Node
        //---------------------------------
        {
            // Extract gradient block for base node of this constraint
            const auto B = Kokkos::subview(
                gradient_terms, i_constraint, Kokkos::ALL,
                Kokkos::make_pair(
                    node_col_range(i_constraint, 0, 0), node_col_range(i_constraint, 0, 1)
                )
            );

            // B(0:3,0:3) = -I
            for (int i = 0; i < 3; ++i) {
                B(i, i) = -1.;
            }

            // B(3,3:6) = cross(R1 * x0_hat, transpose(R2 * z0_hat))
            // B(4,3:6) = cross(R1 * x0_hat, transpose(R2 * y0_hat))
            for (int j = 0; j < 3; ++j) {
                B(3, j + 3) = xcz(j);
                B(4, j + 3) = xcy(j);
            }
        }
    }
};

}  // namespace openturbine
