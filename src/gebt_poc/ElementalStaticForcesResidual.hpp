#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/CalculateSectionalStrain.hpp"
#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/NodalCurvature.hpp"
#include "src/gebt_poc/NodalElasticForces.hpp"
#include "src/gebt_poc/SectionalStiffness.hpp"
#include "src/gebt_poc/interpolation.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void ElementalStaticForcesResidual(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    View2D_6x6::const_type stiffness, const Quadrature& quadrature, View1D residual
) {
    const auto n_nodes = gen_coords.extent(0);
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto nodes = VectorFieldView("nodes", n_nodes);
    Kokkos::deep_copy(
        nodes, Kokkos::subview(position_vectors, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );

    // Allocate Views for some required intermediate variables
    auto gen_coords_qp = View1D_LieGroup("gen_coords_qp");
    auto gen_coords_derivatives_qp = View1D_LieGroup("gen_coords_derivatives_qp");
    auto position_vector_qp = View1D_LieGroup("position_vector_qp");
    auto pos_vector_derivatives_qp = View1D_LieGroup("pos_vector_derivatives_qp");
    auto curvature = View1D_Vector("curvature");
    auto sectional_strain = View1D_LieAlgebra("sectional_strain");
    auto sectional_stiffness = View2D_6x6("sectional_stiffness");

    Kokkos::deep_copy(residual, 0.);
    for (size_t i = 0; i < n_nodes; ++i) {
        const auto node_count = i;
        for (size_t j = 0; j < n_quad_pts; ++j) {
            // Calculate required interpolated values at the quadrature point
            const auto q_pt = quadrature.GetQuadraturePoints()[j];
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

            // Calculate the curvature and sectional strain
            NodalCurvature(gen_coords_qp, gen_coords_derivatives_qp, curvature);
            CalculateSectionalStrain(
                pos_vector_derivatives_qp, gen_coords_derivatives_qp, curvature, sectional_strain
            );

            // Calculate the sectional stiffness matrix in inertial basis
            auto rotation_0 = gen_alpha_solver::EulerParameterToRotationMatrix(
                Kokkos::subview(position_vector_qp, Kokkos::make_pair(3, 7))
            );
            auto rotation = gen_alpha_solver::EulerParameterToRotationMatrix(
                Kokkos::subview(gen_coords_qp, Kokkos::make_pair(3, 7))
            );
            SectionalStiffness(stiffness, rotation_0, rotation, sectional_stiffness);

            // Calculate elastic forces i.e. F^C and F^D vectors
            auto elastic_forces_fc = View1D_LieAlgebra("elastic_forces_fc");
            auto elastic_forces_fd = View1D_LieAlgebra("elastic_forces_fd");
            NodalElasticForces(
                sectional_strain, rotation, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                sectional_stiffness, elastic_forces_fc, elastic_forces_fd
            );

            // Calculate the residual at the quadrature point
            const auto q_weight = quadrature.GetQuadratureWeights()[j];
            Kokkos::parallel_for(
                LieAlgebraComponents,
                KOKKOS_LAMBDA(const size_t component) {
                    residual(node_count * LieAlgebraComponents + component) +=
                        q_weight * (shape_function_derivative_vector(node_count) *
                                        elastic_forces_fc(component) +
                                    jacobian * shape_function_vector(node_count) *
                                        elastic_forces_fd(component));
                }
            );
        }
    }
}

}  // namespace openturbine::gebt_poc