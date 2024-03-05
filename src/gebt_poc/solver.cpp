#include "src/gebt_poc/solver.h"

#include <KokkosBlas.hpp>

#include "src/gebt_poc/element.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/NodalCurvature.hpp"
#include "src/gebt_poc/CalculateSectionalStrain.hpp"

namespace openturbine::gebt_poc {

void SectionalStiffness(
    const StiffnessMatrix& stiffness, View2D_3x3::const_type rotation_0,
    View2D_3x3::const_type rotation, View2D_6x6 sectional_stiffness
) {
    auto total_rotation = View2D_3x3("total_rotation");
    KokkosBlas::gemm("N", "N", 1., rotation, rotation_0, 0., total_rotation);

    // rotation_matrix_6x6 = [
    //    [total_rotation]          [0]_3x3
    //        [0]_3x3           total_rotation
    // ]
    auto rotation_matrix = View2D_6x6("rotation_matrix");
    Kokkos::deep_copy(rotation_matrix, 0.);
    auto rotation_matrix_1 =
        Kokkos::subview(rotation_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(rotation_matrix_1, total_rotation);
    auto rotation_matrix_2 =
        Kokkos::subview(rotation_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(rotation_matrix_2, total_rotation);

    // Calculate the sectional stiffness matrix in inertial basis
    Kokkos::deep_copy(sectional_stiffness, 0.);
    auto stiffness_matrix_left_rot = View2D_6x6("temp");
    KokkosBlas::gemm(
        "N", "N", 1., rotation_matrix, stiffness.GetStiffnessMatrix(), 0., stiffness_matrix_left_rot
    );
    KokkosBlas::gemm(
        "N", "T", 1., stiffness_matrix_left_rot, rotation_matrix, 0., sectional_stiffness
    );
}

void NodalElasticForces(
    View1D_LieAlgebra::const_type sectional_strain, View2D_3x3::const_type rotation,
    View1D_LieGroup::const_type pos_vector_derivatives,
    View1D_LieGroup::const_type gen_coords_derivatives, View2D_6x6::const_type sectional_stiffness,
    View1D_LieAlgebra elastic_forces_fc, View1D_LieAlgebra elastic_forces_fd
) {
    // Calculate first part of the elastic forces i.e. F^C vector
    auto sectional_strain_next = View1D_LieAlgebra("sectional_strain_next");
    Kokkos::deep_copy(sectional_strain_next, sectional_strain);

    auto sectional_strain_next_1 = Kokkos::subview(sectional_strain_next, Kokkos::make_pair(0, 3));
    auto x0_prime = Kokkos::subview(pos_vector_derivatives, Kokkos::make_pair(0, 3));
    auto R_x0_prime = View1D_Vector("R_x0_prime");
    KokkosBlas::gemv("N", -1., rotation, x0_prime, 0., R_x0_prime);
    KokkosBlas::axpy(1., R_x0_prime, sectional_strain_next_1);

    Kokkos::deep_copy(elastic_forces_fc, 0.);
    KokkosBlas::gemv("N", 1., sectional_stiffness, sectional_strain_next, 0., elastic_forces_fc);

    // Calculate second part of the elastic forces i.e. F^D vector
    auto x0_prime_tilde = gen_alpha_solver::create_cross_product_matrix(x0_prime);
    auto u_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(gen_coords_derivatives, Kokkos::make_pair(0, 3))
    );
    auto fd_values = View2D_3x3("fd_values");
    Kokkos::deep_copy(fd_values, x0_prime_tilde);
    KokkosBlas::axpy(1., u_prime_tilde, fd_values);

    Kokkos::deep_copy(elastic_forces_fd, 0.);
    auto elastic_force_fd_1 = Kokkos::subview(elastic_forces_fd, Kokkos::make_pair(3, 6));
    KokkosBlas::gemv(
        "T", 1., fd_values, Kokkos::subview(elastic_forces_fc, Kokkos::make_pair(0, 3)), 0.,
        elastic_force_fd_1
    );
}

void ElementalStaticForcesResidual(
    View1D::const_type position_vectors, View1D::const_type gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, View1D residual
) {
    const auto n_nodes = gen_coords.extent(0) / LieGroupComponents;
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto nodes = VectorFieldView("nodes", n_nodes);
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

void ElementalStaticForcesResidual(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, View1D residual
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

void SectionalMassMatrix(
    const MassMatrix& mass_matrix, View2D_3x3 rotation_0, View2D_3x3 rotation,
    View2D_6x6 sectional_mass_matrix
) {
    auto total_rotation = View2D_3x3("total_rotation");
    KokkosBlas::gemm("N", "N", 1., rotation, rotation_0, 0., total_rotation);

    // rotation_matrix_6x6 = [
    //    [total_rotation]          [0]_3x3
    //        [0]_3x3           total_rotation
    // ]
    auto rotation_matrix = View2D_6x6("rotation_matrix");
    Kokkos::deep_copy(rotation_matrix, 0.);
    auto rotation_matrix_1 =
        Kokkos::subview(rotation_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(rotation_matrix_1, total_rotation);
    auto rotation_matrix_2 =
        Kokkos::subview(rotation_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(rotation_matrix_2, total_rotation);

    // Calculate the sectional mass matrix in inertial basis
    Kokkos::deep_copy(sectional_mass_matrix, 0.);
    auto mass_matrix_left_rot = View2D_6x6("temp");
    KokkosBlas::gemm(
        "N", "N", 1., rotation_matrix, mass_matrix.GetMassMatrix(), 0., mass_matrix_left_rot
    );
    KokkosBlas::gemm("N", "T", 1., mass_matrix_left_rot, rotation_matrix, 0., sectional_mass_matrix);
}

void NodalInertialForces(
    View1D_LieAlgebra::const_type velocity, View1D_LieAlgebra::const_type acceleration,
    const MassMatrix& sectional_mass_matrix, View1D_LieAlgebra inertial_forces_fc
) {
    // The inertial forces vector is defined as
    // {inertial_forces}_6x1 = {
    //     mass * { u_dot_dot + (omega_dot_tilde + omega_tilde * omega_tilde) * eta }_3x1,
    //     { mass * eta_tilde * u_dot_dot + [rho] * omega_dot + omega_tilde * [rho] * omega }_3x1
    // }
    // where,
    // mass - 1x1 = scalar mass of the beam element (from the sectional mass matrix)
    // u_dot_dot - 3x1 = translational acceleration of the center of mass of the beam element
    // omega - 3x1 = angular velocity of the beam element
    // omega_dot - 3x1 = angular acceleration of the beam element
    // omega_tilde - 3x3 = skew symmetric matrix of omega
    // eta - 3x1 = center of mass of the beam element
    // eta_tilde - 3x3 = skew symmetric matrix of eta
    // rho - 3x3 = moment of inertia matrix of the beam element (from the sectional mass matrix)

    Kokkos::deep_copy(inertial_forces_fc, 0.);

    // Calculate mass, {eta}, and [rho] from the sectional mass matrix
    auto mass = sectional_mass_matrix.GetMass();
    auto eta = sectional_mass_matrix.GetCenterOfMass();
    auto rho = sectional_mass_matrix.GetMomentOfInertia();

    // Calculate the first 3 elements of the inertial forces vector
    auto inertial_forces_fc_1 = Kokkos::subview(inertial_forces_fc, Kokkos::make_pair(0, 3));
    auto angular_velocity = Kokkos::subview(velocity, Kokkos::make_pair(3, 6));
    auto angular_velocity_tilde = gen_alpha_solver::create_cross_product_matrix(angular_velocity);
    auto accelaration = Kokkos::subview(acceleration, Kokkos::make_pair(0, 3));
    auto angular_acceleration = Kokkos::subview(acceleration, Kokkos::make_pair(3, 6));
    auto angular_acceleration_tilde =
        gen_alpha_solver::create_cross_product_matrix(angular_acceleration);

    auto temp = View2D_3x3("temp");
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, angular_velocity_tilde, 0., temp);
    KokkosBlas::axpy(1., angular_acceleration_tilde, temp);
    KokkosBlas::gemv("N", mass, temp, eta, 0., inertial_forces_fc_1);
    KokkosBlas::axpy(mass, accelaration, inertial_forces_fc_1);

    // Calculate the last 3 elements of the inertial forces vector
    auto inertial_forces_fc_2 = Kokkos::subview(inertial_forces_fc, Kokkos::make_pair(3, 6));
    auto center_of_mass_tilde = gen_alpha_solver::create_cross_product_matrix(eta);

    KokkosBlas::gemv("N", 1., rho, angular_acceleration, 0., inertial_forces_fc_2);
    auto temp2 = View2D_3x3("temp2");
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, rho, 0., temp2);
    KokkosBlas::gemv("N", 1., temp2, angular_velocity, 1., inertial_forces_fc_2);
    KokkosBlas::gemv("N", mass, center_of_mass_tilde, accelaration, 1., inertial_forces_fc_2);
}

void ElementalInertialForcesResidual(
    View1D::const_type position_vectors, View1D::const_type gen_coords, View1D::const_type velocity,
    View1D::const_type acceleration, const MassMatrix& mass_matrix, const Quadrature& quadrature,
    View1D residual
) {
    const auto n_nodes = gen_coords.extent(0) / LieGroupComponents;
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto nodes = VectorFieldView("nodes", n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        auto index = i * LieGroupComponents;
        Kokkos::deep_copy(
            Kokkos::subview(nodes, i, Kokkos::ALL),
            Kokkos::subview(position_vectors, Kokkos::make_pair(index, index + 3))
        );
    }

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
            const auto q_pt = quadrature.GetQuadraturePoints()[j];
            auto shape_function = LagrangePolynomial(order, q_pt);
            auto shape_function_derivative = LagrangePolynomialDerivative(order, q_pt);
            auto shape_function_vector = gen_alpha_solver::create_vector(shape_function);
            auto shape_function_derivative_vector =
                gen_alpha_solver::create_vector(shape_function_derivative);

            auto jacobian = CalculateJacobian(nodes, shape_function_derivative_vector);
            InterpolateNodalValues(gen_coords, shape_function, gen_coords_qp);
            InterpolateNodalValues(position_vectors, shape_function, position_vector_qp);
            InterpolateNodalValues(velocity, shape_function, velocity_qp, LieAlgebraComponents);
            InterpolateNodalValues(
                acceleration, shape_function, acceleration_qp, LieAlgebraComponents
            );

            // Calculate the sectional mass matrix in inertial basis
            auto rotation_0 = gen_alpha_solver::EulerParameterToRotationMatrix(
                Kokkos::subview(position_vector_qp, Kokkos::make_pair(3, 7))
            );
            auto rotation = gen_alpha_solver::EulerParameterToRotationMatrix(
                Kokkos::subview(gen_coords_qp, Kokkos::make_pair(3, 7))
            );
            SectionalMassMatrix(mass_matrix, rotation_0, rotation, sectional_mass_matrix);

            auto mm = MassMatrix(sectional_mass_matrix);
            NodalInertialForces(velocity_qp, acceleration_qp, mm, inertial_f);

            // Calculate the residual at the quadrature point
            const auto q_weight = quadrature.GetQuadratureWeights()[j];
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

void ElementalInertialForcesResidual(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    LieAlgebraFieldView::const_type velocity, LieAlgebraFieldView::const_type acceleration,
    const MassMatrix& mass_matrix, const Quadrature& quadrature, View1D residual
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
            const auto q_pt = quadrature.GetQuadraturePoints()[j];
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

            auto mm = MassMatrix(sectional_mass_matrix);
            NodalInertialForces(velocity_qp, acceleration_qp, mm, inertial_f);

            // Calculate the residual at the quadrature point
            const auto q_weight = quadrature.GetQuadratureWeights()[j];
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

void CalculateOMatrix(
    View2D::const_type N_tilde, View2D::const_type M_tilde, View2D::const_type C11,
    View2D::const_type C21, View2D::const_type values, View2D O_matrix
) {
    // non_zero_terms_part_1 = -N_tilde + [C11] * (x_0_prime_tilde + u_prime_tilde)
    auto non_zero_terms_part_1 = View2D_3x3("non_zero_terms_part_1");
    KokkosBlas::gemm("N", "N", 1., C11, values, 0., non_zero_terms_part_1);
    KokkosBlas::axpy(-1., N_tilde, non_zero_terms_part_1);

    // non_zero_terms_part_2 = -M_tilde + [C21] * (x_0_prime_tilde  + u_prime_tilde)
    auto non_zero_terms_part_2 = View2D_3x3("non_zero_terms_part_2");
    KokkosBlas::gemm("N", "N", 1., C21, values, 0., non_zero_terms_part_2);
    KokkosBlas::axpy(-1., M_tilde, non_zero_terms_part_2);

    // Assemble the O matrix
    // [O]_6x6 = [
    //     [0]_3x3      -N_tilde + [C11] * (x_0_prime_tilde + u_prime_tilde)
    //     [0]_3x3      -M_tilde + [C21] * (x_0_prime_tilde  + u_prime_tilde)
    // ]
    Kokkos::deep_copy(O_matrix, 0.);
    auto O_matrix_1 = Kokkos::subview(O_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(O_matrix_1, non_zero_terms_part_1);
    auto O_matrix_2 = Kokkos::subview(O_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(O_matrix_2, non_zero_terms_part_2);
}

void CalculatePMatrix(
    View2D::const_type N_tilde, View2D::const_type C11, View2D::const_type C12,
    View2D::const_type values, View2D P_matrix
) {
    // non_zero_terms_part_3 = (x_0_prime_tilde + u_prime_tilde)^T * [C11]
    auto non_zero_terms_part_3 = View2D_3x3("non_zero_terms_part_3");
    KokkosBlas::gemm("T", "N", 1., values, C11, 0., non_zero_terms_part_3);
    KokkosBlas::axpy(1., N_tilde, non_zero_terms_part_3);

    // non_zero_terms_part_4 = (x_0_prime_tilde  + u_prime_tilde)^T * [C12]
    auto non_zero_terms_part_4 = View2D_3x3("non_zero_terms_part_4");
    KokkosBlas::gemm("T", "N", 1., values, C12, 0., non_zero_terms_part_4);

    // Assemble the P matrix
    // [P]_6x6 = [
    //                         [0]_3x3                                      [0]_3x3
    //     N_tilde + (x_0_prime_tilde + u_prime_tilde)^T * [C11]     (x_0_prime_tilde  +
    //     u_prime_tilde)^T * [C12]
    // ]
    Kokkos::deep_copy(P_matrix, 0.);
    auto P_matrix_1 = Kokkos::subview(P_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(P_matrix_1, non_zero_terms_part_3);
    auto P_matrix_2 = Kokkos::subview(P_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(P_matrix_2, non_zero_terms_part_4);
}

void CalculateQMatrix(
    View2D::const_type N_tilde, View2D::const_type C11, View2D::const_type values, View2D Q_matrix
) {
    // Assemble the Q matrix
    // [Q]_6x6 = [
    //     [0]_3x3                                          [0]_3x3
    //     [0]_3x3      (x_0_prime_tilde + u_prime_tilde)^T * (-N_tilde + [C11] * (x_0_prime_tilde  +
    //     u_prime_tilde))
    // ]
    auto val = View2D_3x3("val");
    KokkosBlas::gemm("N", "N", 1., C11, values, 0., val);
    KokkosBlas::axpy(-1., N_tilde, val);

    Kokkos::deep_copy(Q_matrix, 0.);
    auto non_zero_term = Kokkos::subview(Q_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    KokkosBlas::gemm("T", "N", 1., values, val, 0., non_zero_term);
}

void NodalStaticStiffnessMatrixComponents(
    View1D_LieAlgebra::const_type elastic_force_fc,
    View1D_LieGroup::const_type pos_vector_derivatives,
    View1D_LieGroup::const_type gen_coords_derivatives, View2D_6x6::const_type sectional_stiffness,
    View2D_6x6 O_matrix, View2D_6x6 P_matrix, View2D_6x6 Q_matrix
) {
    auto x0_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(pos_vector_derivatives, Kokkos::make_pair(0, 3))
    );
    auto u_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(gen_coords_derivatives, Kokkos::make_pair(0, 3))
    );

    auto values = View2D_3x3("values");
    Kokkos::deep_copy(values, x0_prime_tilde);
    KokkosBlas::axpy(1., u_prime_tilde, values);

    auto N = Kokkos::subview(elastic_force_fc, Kokkos::make_pair(0, 3));
    auto N_tilde = gen_alpha_solver::create_cross_product_matrix(N);
    auto M = Kokkos::subview(elastic_force_fc, Kokkos::make_pair(3, 6));
    auto M_tilde = gen_alpha_solver::create_cross_product_matrix(M);

    auto C11 =
        Kokkos::subview(sectional_stiffness, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    auto C12 =
        Kokkos::subview(sectional_stiffness, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    auto C21 =
        Kokkos::subview(sectional_stiffness, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));

    // Calculate the O, P, and Q matrices
    CalculateOMatrix(N_tilde, M_tilde, C11, C21, values, O_matrix);
    CalculatePMatrix(N_tilde, C11, C12, values, P_matrix);
    CalculateQMatrix(N_tilde, C11, values, Q_matrix);
}

void ElementalStaticStiffnessMatrix(
    View1D::const_type position_vectors, View1D::const_type gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, View2D stiffness_matrix
) {
    const auto n_nodes = gen_coords.extent(0) / LieGroupComponents;
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto nodes = VectorFieldView("nodes", n_nodes);
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

void ElementalStaticStiffnessMatrix(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, View2D stiffness_matrix
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

void NodalGyroscopicMatrix(
    View1D_LieAlgebra::const_type velocity, const MassMatrix& sectional_mass_matrix,
    View2D_6x6 gyroscopic_matrix
) {
    // The Gyroscopic matrix is defined as
    // {gyroscopic_matrix}_6x6 = [
    //     [0]_3x3      ~[omega_tilde * mass * eta]^T + omega_tilde * mass * eta_tilde^T
    //     [0]_3x3                  omega_tilde * rho - ~[rho * omega]
    // ]
    // where,
    // mass - 1x1 = scalar mass of the beam element (from the sectional mass matrix)
    // eta - 3x1 = center of mass of the beam element
    // omega - 3x1 = angular velocity of the beam element
    // omega_tilde - 3x3 = skew symmetric matrix of omega
    // eta_tilde - 3x3 = skew symmetric matrix of eta
    // rho - 3x3 = moment of inertia matrix of the beam element (from the sectional mass matrix)

    Kokkos::deep_copy(gyroscopic_matrix, 0.);

    // Calculate mass, {eta}, and [rho] from the sectional mass matrix
    auto mass = sectional_mass_matrix.GetMass();
    auto eta = sectional_mass_matrix.GetCenterOfMass();
    auto rho = sectional_mass_matrix.GetMomentOfInertia();

    // Calculate the top right block i.e. quadrant 1 of the gyroscopic matrix
    auto gyroscopic_matrix_q1 =
        Kokkos::subview(gyroscopic_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    auto angular_velocity = Kokkos::subview(velocity, Kokkos::make_pair(3, 6));
    auto angular_velocity_tilde = gen_alpha_solver::create_cross_product_matrix(angular_velocity);
    auto center_of_mass_tilde = gen_alpha_solver::create_cross_product_matrix(eta);

    auto temp1 = Kokkos::View<double[3]>("temp1");
    KokkosBlas::gemv("N", mass, angular_velocity_tilde, eta, 0., temp1);
    auto gyroscopic_matrix_q1_part1 =
        gen_alpha_solver::transpose_matrix(gen_alpha_solver::create_cross_product_matrix(temp1));
    KokkosBlas::gemm(
        "N", "T", mass, angular_velocity_tilde, center_of_mass_tilde, 0., gyroscopic_matrix_q1
    );
    KokkosBlas::axpy(1., gyroscopic_matrix_q1_part1, gyroscopic_matrix_q1);

    // Calculate the bottom right block i.e. quadrant 4 of the gyroscopic matrix
    auto gyroscopic_matrix_q4 =
        Kokkos::subview(gyroscopic_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, rho, 0., gyroscopic_matrix_q4);
    auto temp2 = Kokkos::View<double[3]>("temp2");
    KokkosBlas::gemv("N", 1., rho, angular_velocity, 1., temp2);
    auto gyroscopic_matrix_q4_part2 = gen_alpha_solver::create_cross_product_matrix(temp2);
    KokkosBlas::axpy(-1., gyroscopic_matrix_q4_part2, gyroscopic_matrix_q4);
}

void NodalDynamicStiffnessMatrix(
    View1D_LieAlgebra::const_type velocity, View1D_LieAlgebra::const_type acceleration,
    const MassMatrix& sectional_mass_matrix, View2D_6x6 stiffness_matrix
) {
    // The dynamic stiffness matrix is defined as
    // {dyn_stiffness_matrix}_6x6 = [
    //     [0]_3x3      (omega_dot_tilde + omega_tilde * omega_tilde) * mass * eta_tilde^T
    //
    //     [0]_3x3               acceleration_tilde * mass * eta_tilde + (rho * omega_dot_tilde  -
    //                      ~[rho * omega_dot]) + omega_tilde * (rho * omega_tilde - ~[rho * omega])
    // ]
    // where,
    // mass - 1x1 = scalar mass of the beam element (from the sectional mass matrix)
    // u_dot_dot - 3x1 = translational acceleration of the center of mass of the beam element
    // omega - 3x1 = angular velocity of the beam element
    // omega_dot - 3x1 = angular acceleration of the beam element
    // omega_tilde - 3x3 = skew symmetric matrix of omega
    // eta - 3x1 = center of mass of the beam element
    // eta_tilde - 3x3 = skew symmetric matrix of eta
    // rho - 3x3 = moment of inertia matrix of the beam element (from the sectional mass matrix)

    Kokkos::deep_copy(stiffness_matrix, 0.);

    // Calculate mass, {eta}, and [rho] from the sectional mass matrix
    auto mass = sectional_mass_matrix.GetMass();
    auto eta = sectional_mass_matrix.GetCenterOfMass();
    auto rho = sectional_mass_matrix.GetMomentOfInertia();

    // Calculate the top right block i.e. quadrant 1 of the dynamic stiffness matrix
    auto stiffness_matrix_q1 =
        Kokkos::subview(stiffness_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    auto angular_velocity = Kokkos::subview(velocity, Kokkos::make_pair(3, 6));
    auto angular_velocity_tilde = gen_alpha_solver::create_cross_product_matrix(angular_velocity);
    auto angular_acceleration = Kokkos::subview(acceleration, Kokkos::make_pair(3, 6));
    auto angular_acceleration_tilde =
        gen_alpha_solver::create_cross_product_matrix(angular_acceleration);
    auto center_of_mass_tilde = gen_alpha_solver::create_cross_product_matrix(eta);

    auto temp1 = View2D_3x3("temp1");
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, angular_velocity_tilde, 0., temp1);
    KokkosBlas::axpy(1., angular_acceleration_tilde, temp1);
    KokkosBlas::gemm("N", "T", mass, temp1, center_of_mass_tilde, 0., stiffness_matrix_q1);

    // Calculate the bottom right block i.e. quadrant 4 of the dynamic stiffness matrix
    auto stiffness_matrix_q4 =
        Kokkos::subview(stiffness_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    auto accelaration = Kokkos::subview(acceleration, Kokkos::make_pair(0, 3));
    auto accelaration_tilde = gen_alpha_solver::create_cross_product_matrix(accelaration);

    // part 1: acceleration_tilde * mass * eta_tilde
    auto temp2 = View2D_3x3("temp2");
    KokkosBlas::gemm("N", "N", mass, accelaration_tilde, center_of_mass_tilde, 0., temp2);
    // part 2: (rho * omega_dot_tilde  - ~[rho * omega_dot])
    auto temp3 = View2D_3x3("temp3");
    KokkosBlas::gemm("N", "N", 1., rho, angular_acceleration_tilde, 0., temp3);
    auto temp4 = View1D_Vector("temp4");
    KokkosBlas::gemv("N", 1., rho, angular_acceleration, 1., temp4);
    auto temp5 = gen_alpha_solver::create_cross_product_matrix(temp4);
    KokkosBlas::axpy(-1., temp5, temp3);
    // part 3: omega_tilde * (rho * omega_tilde - ~[rho * omega])
    auto temp6 = View2D_3x3("temp6");
    KokkosBlas::gemm("N", "N", 1., rho, angular_velocity_tilde, 0., temp6);
    auto temp7 = View1D_Vector("temp7");
    KokkosBlas::gemv("N", 1., rho, angular_velocity, 1., temp7);
    auto temp8 = gen_alpha_solver::create_cross_product_matrix(temp7);
    KokkosBlas::axpy(-1., temp8, temp6);
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, temp6, 0., stiffness_matrix_q4);

    KokkosBlas::axpy(1., temp2, stiffness_matrix_q4);
    KokkosBlas::axpy(1., temp3, stiffness_matrix_q4);
}

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
