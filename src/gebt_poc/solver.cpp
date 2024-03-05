#include "src/gebt_poc/solver.h"

#include <KokkosBlas.hpp>

#include "src/gebt_poc/element.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/NodalCurvature.hpp"
#include "src/gebt_poc/CalculateSectionalStrain.hpp"
#include "src/gebt_poc/SectionalStiffness.hpp"
#include "src/gebt_poc/NodalElasticForces.hpp"
#include "src/gebt_poc/SectionalMassMatrix.hpp"
#include "src/gebt_poc/NodalInertialForces.hpp"
#include "src/gebt_poc/NodalStaticStiffnessMatrixComponents.hpp"
#include "src/gebt_poc/NodalGyroscopicMatrix.hpp"
#include "src/gebt_poc/NodalDynamicStiffnessMatrix.hpp"

namespace openturbine::gebt_poc {

void ElementalInertialMatrices(
    View1D::const_type position_vectors, View1D::const_type gen_coords, View1D::const_type velocity,
    View1D::const_type acceleration, const MassMatrix& mass_matrix, const Quadrature& quadrature,
    View2D element_mass_matrix, View2D element_gyroscopic_matrix,
    View2D element_dynamic_stiffness_matrix
) {
    const auto n_nodes = gen_coords.extent(0) / LieGroupComponents;
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto nodes = Kokkos::View<double* [3]>("nodes", n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        auto index = i * LieGroupComponents;
        Kokkos::deep_copy(
            Kokkos::subview(nodes, i, Kokkos::ALL),
            Kokkos::subview(position_vectors, Kokkos::make_pair(index, index + 3))
        );
    }

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
        InterpolateNodalValues(velocity, shape_function, velocity_qp, LieAlgebraComponents);
        InterpolateNodalValues(acceleration, shape_function, acceleration_qp, LieAlgebraComponents);

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

void ElementalInertialMatrices(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    LieAlgebraFieldView::const_type velocity, LieAlgebraFieldView::const_type acceleration,
    const MassMatrix& mass_matrix, const Quadrature& quadrature, View2D element_mass_matrix,
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

void ElementalConstraintForcesResidual(View1D::const_type gen_coords, View1D constraints_residual) {
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
                    gen_coords(3), gen_coords(4), gen_coords(5), gen_coords(6)
                )
            );
            // Set residual as translation and rotation of root node
            // TODO: update when position & rotations are prescribed
            constraints_residual(0) = gen_coords(0);
            constraints_residual(1) = gen_coords(1);
            constraints_residual(2) = gen_coords(2);
            constraints_residual(3) = rotation_vector.GetXComponent();
            constraints_residual(4) = rotation_vector.GetYComponent();
            constraints_residual(5) = rotation_vector.GetZComponent();
        }
    );
}

void ElementalConstraintForcesResidual(
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

void ElementalConstraintForcesGradientMatrix(
    View1D::const_type gen_coords, View1D::const_type position_vector,
    View2D constraints_gradient_matrix
) {
    auto translation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(0, 3));
    auto rotation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(3, 7));
    auto rotation_matrix_0 = gen_alpha_solver::EulerParameterToRotationMatrix(rotation_0);
    auto position_0 = Kokkos::subview(position_vector, Kokkos::make_pair(0, 3));

    // position_cross_prod_matrix = ~{position_0} + ~{translation_0}
    auto position_0_cross_prod_matrix = gen_alpha_solver::create_cross_product_matrix(position_0);
    auto translation_0_cross_prod_matrix =
        gen_alpha_solver::create_cross_product_matrix(translation_0);
    auto position_cross_prod_matrix = View2D_3x3("position_cross_prod_matrix");
    Kokkos::deep_copy(position_cross_prod_matrix, position_0_cross_prod_matrix);
    KokkosBlas::axpy(1., translation_0_cross_prod_matrix, position_cross_prod_matrix);

    // Assemble the constraint gradient matrix i.e. B matrix for the beam element
    // [B]_6x(n+1) = [
    //     [B11]_3x3              0            0   ....  0
    //     [B21]_3x3          [B22]_3x3        0   ....  0
    // ]
    // where
    // [B11]_3x3 = [1]_3x3
    // [B21]_3x3 = -[rotation_matrix_0]_3x3
    // [B22]_3x3 = -[rotation_matrix_0]_3x3 * [position_cross_prod_matrix]_3x3
    // n = order of the element
    Kokkos::deep_copy(constraints_gradient_matrix, 0.);
    auto B11 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3)
    );
    auto B21 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3)
    );
    auto B22 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6)
    );

    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            B11(0, 0) = 1.;
            B11(1, 1) = 1.;
            B11(2, 2) = 1.;
        }
    );

    KokkosBlas::scal(B21, -1., rotation_matrix_0);
    KokkosBlas::gemm("N", "N", -1., rotation_matrix_0, position_cross_prod_matrix, 0., B22);
}

}  // namespace openturbine::gebt_poc
