#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/CalculateSectionalStrain.hpp"
#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/NodalCurvature.hpp"
#include "src/gebt_poc/NodalElasticForces.hpp"
#include "src/gebt_poc/NodalStaticStiffnessMatrixComponents.hpp"
#include "src/gebt_poc/SectionalStiffness.hpp"
#include "src/gebt_poc/interpolation.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

inline void ElementalStaticStiffnessMatrix(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    View2D_6x6::const_type stiffness, const Quadrature& quadrature, View2D stiffness_matrix
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
    auto O_matrix = View2D_6x6("O_matrix");
    auto P_matrix = View2D_6x6("P_matrix");
    auto Q_matrix = View2D_6x6("Q_matrix");

    Kokkos::deep_copy(stiffness_matrix, 0.);
    for (size_t i = 0; i < n_nodes; ++i) {
        for (size_t j = 0; j < n_nodes; ++j) {
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

                // Calculate the stiffness matrix components, i.e. O, P, and Q matrices
                NodalStaticStiffnessMatrixComponents(
                    elastic_forces_fc, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                    sectional_stiffness, O_matrix, P_matrix, Q_matrix
                );

                const auto q_weight = quadrature.GetQuadratureWeights()[k];
                Kokkos::parallel_for(
                    Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
                        {0, 0}, {LieAlgebraComponents, LieAlgebraComponents}
                    ),
                    KOKKOS_LAMBDA(const size_t ii, const size_t jj) {
                        stiffness_matrix(
                            i * LieAlgebraComponents + ii, j * LieAlgebraComponents + jj
                        ) += q_weight *
                             (shape_function_vector(i) * P_matrix(ii, jj) *
                                  shape_function_derivative_vector(j) +
                              shape_function_vector(i) * Q_matrix(ii, jj) *
                                  shape_function_vector(j) * jacobian +
                              shape_function_derivative_vector(i) * sectional_stiffness(ii, jj) *
                                  shape_function_derivative_vector(j) / jacobian +
                              shape_function_derivative_vector(i) * O_matrix(ii, jj) *
                                  shape_function_vector(j));
                    }
                );
            }
        }
    }
}
}  // namespace openturbine::gebt_poc