#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/element.h"
#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/NodalInertialForces.hpp"
#include "src/gebt_poc/SectionalMassMatrix.hpp"
#include "src/gebt_poc/interpolation.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/quaternion.h"

namespace openturbine::gebt_poc {

inline void ElementalInertialForcesResidual(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    LieAlgebraFieldView::const_type velocity, LieAlgebraFieldView::const_type acceleration,
    View2D_6x6::const_type mass_matrix, const Quadrature& quadrature, View1D residual
) {
    const auto n_nodes = gen_coords.extent(0);
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.points.extent(0);

    auto nodes = VectorFieldView("nodes", n_nodes);
    Kokkos::deep_copy(
        nodes, Kokkos::subview(position_vectors, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );

    // Allocate Views for some required intermediate variables
    auto gen_coords_qp = View1D_LieGroup("gen_coords_qp");
    auto position_vector_qp = View1D_LieGroup("position_vector_qp");
    auto velocity_qp = View1D_LieAlgebra("velocity_qp");
    auto acceleration_qp = View1D_LieAlgebra("acceleration_qp");
    auto sectional_mass_matrix = View2D_6x6("sectional_mass_matrix");
    auto inertial_f = View1D_LieAlgebra("inertial_f");

    Kokkos::deep_copy(residual, 0.);
    for (size_t i = 0; i < n_nodes; ++i) {
        const auto node_count = i;
        for (size_t j = 0; j < n_quad_pts; ++j) {
            // Calculate required interpolated values at the quadrature point
            const auto q_pt = quadrature.points(j);
            auto shape_function = LagrangePolynomial(order, q_pt);
            auto shape_function_derivative = LagrangePolynomialDerivative(order, q_pt);
            auto shape_function_vector = gen_alpha_solver::create_vector(shape_function);
            auto shape_function_derivative_vector =
                gen_alpha_solver::create_vector(shape_function_derivative);

            auto jacobian = CalculateJacobian(nodes, shape_function_derivative_vector);
            InterpolateNodalValues(gen_coords, shape_function, gen_coords_qp);
            InterpolateNodalValues(position_vectors, shape_function, position_vector_qp);
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

            NodalInertialForces(velocity_qp, acceleration_qp, sectional_mass_matrix, inertial_f);

            // Calculate the residual at the quadrature point
            const auto q_weight = quadrature.weights(j);
            Kokkos::parallel_for(
                LieAlgebraComponents,
                KOKKOS_LAMBDA(const size_t component) {
                    residual(node_count * LieAlgebraComponents + component) +=
                        q_weight *
                        (jacobian * shape_function_vector(node_count) * inertial_f(component));
                }
            );
        }
    }
}

}  // namespace openturbine::gebt_poc