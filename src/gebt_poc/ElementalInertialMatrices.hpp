#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/NodalDynamicStiffnessMatrix.hpp"
#include "src/gebt_poc/NodalGyroscopicMatrix.hpp"
#include "src/gebt_poc/SectionalMassMatrix.hpp"
#include "src/gebt_poc/interpolation.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

inline void ElementalInertialMatrices(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    LieAlgebraFieldView::const_type velocity, LieAlgebraFieldView::const_type acceleration,
    View2D_6x6::const_type mass_matrix, const Quadrature& quadrature, View2D element_mass_matrix,
    View2D element_gyroscopic_matrix, View2D element_dynamic_stiffness_matrix
) {
    const auto n_nodes = gen_coords.extent(0);
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto nodes = Kokkos::View<double* [3]>("nodes", n_nodes);
    Kokkos::deep_copy(
        nodes, Kokkos::subview(position_vectors, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );

    // Allocate Views for some required intermediate variables
    auto gen_coords_qp = View1D_LieGroup("gen_coords_qp");
    auto gen_coords_derivatives_qp = View1D_LieGroup("gen_coords_derivatives_qp");
    auto position_vector_qp = View1D_LieGroup("position_vector_qp");
    auto pos_vector_derivatives_qp = View1D_LieGroup("pos_vector_derivatives_qp");
    auto velocity_qp = View1D_LieAlgebra("velocity_qp");
    auto acceleration_qp = View1D_LieAlgebra("acceleration_qp");
    auto sectional_mass_matrix = View2D_6x6("sectional_mass_matrix");

    Kokkos::deep_copy(element_mass_matrix, 0.);

    for (size_t k = 0; k < n_quad_pts; ++k) {
        // Calculate required interpolated values at the quadrature point
        const auto q_pt = quadrature.GetQuadraturePoints()[k];
        auto shape_function = LagrangePolynomial(order, q_pt);
        auto shape_function_derivative = LagrangePolynomialDerivative(order, q_pt);
        auto shape_function_vector = gen_alpha_solver::create_vector(shape_function);
        auto shape_function_derivative_vector =
            gen_alpha_solver::create_vector(shape_function_derivative);

        auto jacobian = CalculateJacobian(nodes, shape_function_derivative_vector);
        InterpolateNodalValues(gen_coords, shape_function, gen_coords_qp);
        InterpolateNodalValueDerivatives(
            gen_coords, shape_function_derivative, jacobian, gen_coords_derivatives_qp
        );
        InterpolateNodalValues(position_vectors, shape_function, position_vector_qp);
        InterpolateNodalValueDerivatives(
            position_vectors, shape_function_derivative, jacobian, pos_vector_derivatives_qp
        );
        InterpolateNodalValues(velocity, shape_function, velocity_qp);
        InterpolateNodalValues(acceleration, shape_function, acceleration_qp);

        // Calculate the sectional mass matrix in inertial basis
        auto rotation_0 = gen_alpha_solver::EulerParameterToRotationMatrix(
            Kokkos::subview(position_vector_qp, Kokkos::make_pair(3, 7))
        );
        auto rotation = gen_alpha_solver::EulerParameterToRotationMatrix(
            Kokkos::subview(gen_coords_qp, Kokkos::make_pair(3, 7))
        );
        SectionalMassMatrix(mass_matrix, rotation_0, rotation, sectional_mass_matrix);

        // Calculate the gyroscopic matrix in inertial basis
        auto gyroscopic_matrix = View2D_6x6("gyroscopic_matrix");
        NodalGyroscopicMatrix(velocity_qp, sectional_mass_matrix, gyroscopic_matrix);

        // Calculate the dynamic stiffness matrix in inertial basis
        auto dynamic_stiffness_matrix = View2D_6x6("dynamic_stiffness_matrix");
        NodalDynamicStiffnessMatrix(
            velocity_qp, acceleration_qp, sectional_mass_matrix, dynamic_stiffness_matrix
        );

        const auto q_weight = quadrature.GetQuadratureWeights()[k];
        for (size_t i = 0; i < n_nodes; ++i) {
            for (size_t j = 0; j < n_nodes; ++j) {
                const auto pair6 = Kokkos::make_pair(0, 6);
                const auto pair_i =
                    Kokkos::make_pair(i * LieAlgebraComponents, (i + 1) * LieAlgebraComponents);
                const auto pair_j =
                    Kokkos::make_pair(j * LieAlgebraComponents, (j + 1) * LieAlgebraComponents);
                const auto a = q_weight * shape_function[i] * shape_function[j] * jacobian;
                KokkosBlas::axpy(
                    a, Kokkos::subview(sectional_mass_matrix, pair6, pair6),
                    Kokkos::subview(element_mass_matrix, pair_i, pair_j)
                );
                KokkosBlas::axpy(
                    a, Kokkos::subview(gyroscopic_matrix, pair6, pair6),
                    Kokkos::subview(element_gyroscopic_matrix, pair_i, pair_j)
                );
                KokkosBlas::axpy(
                    a, Kokkos::subview(dynamic_stiffness_matrix, pair6, pair6),
                    Kokkos::subview(element_dynamic_stiffness_matrix, pair_i, pair_j)
                );
            }
        }
    }
}
}  // namespace openturbine::gebt_poc