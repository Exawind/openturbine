#include "src/gebt_poc/solver.h"

#include <KokkosBlas.hpp>

#include "src/gebt_poc/element.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

void InterpolateNodalValues(
    Kokkos::View<double**> nodal_values, std::vector<double> interpolation_function,
    Kokkos::View<double*> interpolated_values
) {
    const auto n_nodes = nodal_values.extent(0);
    KokkosBlas::fill(interpolated_values, 0.);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        KokkosBlas::axpy(
            interpolation_function[i], Kokkos::subview(nodal_values, i, Kokkos::ALL),
            interpolated_values
        );
    }
    // Normalize the rotation quaternion
    auto q = Kokkos::subview(interpolated_values, Kokkos::pair(3, 7));
    auto norm = KokkosBlas::nrm2(q);
    if (norm != 0.0) {
        KokkosBlas::scal(q, 1. / norm, q);
    }
}

void InterpolateNodalValueDerivatives(
    Kokkos::View<double**> nodal_values, std::vector<double> interpolation_function,
    const double jacobian, Kokkos::View<double*> interpolated_values
) {
    if (jacobian == 0.) {
        throw std::invalid_argument("jacobian must be nonzero");
    }
    const auto n_nodes = nodal_values.extent(0);
    KokkosBlas::fill(interpolated_values, 0.);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        KokkosBlas::axpy(
            interpolation_function[i], Kokkos::subview(nodal_values, i, Kokkos::ALL),
            interpolated_values
        );
    }
    KokkosBlas::scal(interpolated_values, 1. / jacobian, interpolated_values);
}

void CalculateCurvature(
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> gen_coords_derivative,
    Kokkos::View<double*> curvature
) {
    // curvature = B * q_prime
    auto b_matrix = Kokkos::View<double[3][4]>("b_matrix");
    gen_alpha_solver::BMatrixForQuaternions(
        b_matrix, Kokkos::subview(gen_coords, Kokkos::make_pair(3, 7))
    );
    auto q_prime = Kokkos::subview(gen_coords_derivative, Kokkos::make_pair(3, 7));
    KokkosBlas::gemv("N", 2., b_matrix, q_prime, 0., curvature);
}

void CalculateSectionalStrain(
    Kokkos::View<double*> pos_vector_derivatives_qp, Kokkos::View<double*> gen_coords_derivatives_qp,
    Kokkos::View<double*> curvature, Kokkos::View<double*> sectional_strain
) {
    // Calculate the sectional strain based on Eq. (35) in the "SO(3)-based GEBT Beam" document
    // in theory guide
    auto sectional_strain_1 = Kokkos::subview(sectional_strain, Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(
        sectional_strain_1, Kokkos::subview(pos_vector_derivatives_qp, Kokkos::make_pair(0, 3))
    );
    KokkosBlas::axpy(
        1., Kokkos::subview(gen_coords_derivatives_qp, Kokkos::make_pair(0, 3)), sectional_strain_1
    );
    auto sectional_strain_2 = Kokkos::subview(sectional_strain, Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(sectional_strain_2, curvature);
}

void CalculateSectionalStiffness(
    const StiffnessMatrix& stiffness, Kokkos::View<double**> rotation_0,
    Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]> rotation,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness
) {
    auto total_rotation =
        Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>("total_rotation");
    KokkosBlas::gemm("N", "N", 1., rotation, rotation_0, 0., total_rotation);

    // rotation_matrix_6x6 = [
    //    [total_rotation]          [0]_3x3
    //        [0]_3x3           total_rotation
    // ]
    auto rotation_matrix =
        Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>(
            "rotation_matrix"
        );
    Kokkos::deep_copy(rotation_matrix, 0.);
    auto rotation_matrix_1 =
        Kokkos::subview(rotation_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(rotation_matrix_1, total_rotation);
    auto rotation_matrix_2 =
        Kokkos::subview(rotation_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(rotation_matrix_2, total_rotation);

    // Calculate the sectional stiffness matrix in inertial basis
    Kokkos::deep_copy(sectional_stiffness, 0.);
    auto stiffness_matrix_left_rot =
        Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>("temp");
    KokkosBlas::gemm(
        "N", "N", 1., rotation_matrix, stiffness.GetStiffnessMatrix(), 0., stiffness_matrix_left_rot
    );
    KokkosBlas::gemm(
        "N", "T", 1., stiffness_matrix_left_rot, rotation_matrix, 0., sectional_stiffness
    );
}

void CalculateElasticForces(
    const Kokkos::View<double*> sectional_strain, Kokkos::View<double**> rotation,
    const Kokkos::View<double*> pos_vector_derivatives,
    const Kokkos::View<double*> gen_coords_derivatives,
    const Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness,
    Kokkos::View<double[kNumberOfLieGroupComponents]> elastic_forces_fc,
    Kokkos::View<double[kNumberOfLieGroupComponents]> elastic_forces_fd
) {
    // Calculate first part of the elastic forces i.e. F^C vector
    auto sectional_strain_next =
        Kokkos::View<double[kNumberOfLieGroupComponents]>("sectional_strain_next");
    Kokkos::deep_copy(sectional_strain_next, sectional_strain);

    auto sectional_strain_next_1 = Kokkos::subview(sectional_strain_next, Kokkos::make_pair(0, 3));
    auto x0_prime = Kokkos::subview(pos_vector_derivatives, Kokkos::make_pair(0, 3));
    auto R_x0_prime = Kokkos::View<double[kNumberOfVectorComponents]>("R_x0_prime");
    KokkosBlas::gemv("N", -1., rotation, x0_prime, 0., R_x0_prime);
    KokkosBlas::axpy(1., R_x0_prime, sectional_strain_next_1);

    Kokkos::deep_copy(elastic_forces_fc, 0.);
    KokkosBlas::gemv("N", 1., sectional_stiffness, sectional_strain_next, 0., elastic_forces_fc);

    // Calculate second part of the elastic forces i.e. F^D vector
    auto x0_prime_tilde = gen_alpha_solver::create_cross_product_matrix(x0_prime);
    auto u_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(gen_coords_derivatives, Kokkos::make_pair(0, 3))
    );
    auto fd_values =
        Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>("fd_values");
    Kokkos::deep_copy(fd_values, x0_prime_tilde);
    KokkosBlas::axpy(1., u_prime_tilde, fd_values);

    Kokkos::deep_copy(elastic_forces_fd, 0.);
    auto elastic_force_fd_1 = Kokkos::subview(elastic_forces_fd, Kokkos::make_pair(3, 6));
    KokkosBlas::gemv(
        "T", 1., fd_values, Kokkos::subview(elastic_forces_fc, Kokkos::make_pair(0, 3)), 0.,
        elastic_force_fd_1
    );
}

void CalculateStaticResidual(
    const Kokkos::View<double**> position_vectors, const Kokkos::View<double**> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, Kokkos::View<double*> residual
) {
    const auto n_nodes = gen_coords.extent(0);
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto nodes = Kokkos::subview(position_vectors, Kokkos::ALL, Kokkos::make_pair(0, 3));

    // Allocate Views for some required intermediate variables
    auto gen_coords_qp = Kokkos::View<double[kNumberOfLieAlgebraComponents]>("gen_coords_qp");
    auto gen_coords_derivatives_qp =
        Kokkos::View<double[kNumberOfLieAlgebraComponents]>("gen_coords_derivatives_qp");
    auto position_vector_qp =
        Kokkos::View<double[kNumberOfLieAlgebraComponents]>("position_vector_qp");
    auto pos_vector_derivatives_qp =
        Kokkos::View<double[kNumberOfLieAlgebraComponents]>("pos_vector_derivatives_qp");
    auto curvature = Kokkos::View<double[kNumberOfVectorComponents]>("curvature");
    auto sectional_strain = Kokkos::View<double[kNumberOfLieGroupComponents]>("sectional_strain");
    auto sectional_stiffness =
        Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>(
            "sectional_stiffness"
        );

    Kokkos::deep_copy(residual, 0.);
    for (size_t node = 0; node < n_nodes; ++node) {
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
            CalculateCurvature(gen_coords_qp, gen_coords_derivatives_qp, curvature);
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
            CalculateSectionalStiffness(stiffness, rotation_0, rotation, sectional_stiffness);

            // Calculate elastic forces i.e. F^C and F^D vectors
            auto elastic_forces_fc =
                Kokkos::View<double[kNumberOfLieGroupComponents]>("elastic_forces_fc");
            auto elastic_forces_fd =
                Kokkos::View<double[kNumberOfLieGroupComponents]>("elastic_forces_fd");
            CalculateElasticForces(
                sectional_strain, rotation, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                sectional_stiffness, elastic_forces_fc, elastic_forces_fd
            );

            // Calculate the residual at the quadrature point
            const auto q_weight = quadrature.GetQuadratureWeights()[j];
            Kokkos::parallel_for(
                kNumberOfLieGroupComponents,
                KOKKOS_LAMBDA(const size_t i) {
                    residual(node * kNumberOfLieGroupComponents + i) +=
                        q_weight * (shape_function_derivative_vector(node) * elastic_forces_fc(i) +
                                    jacobian * shape_function_vector(node) * elastic_forces_fd(i));
                }
            );
        }
    }
}

void CalculateOMatrix(
    const Kokkos::View<double**> N_tilde, const Kokkos::View<double**> M_tilde,
    const Kokkos::View<double**> C11, const Kokkos::View<double**> C21,
    const Kokkos::View<double**> values, Kokkos::View<double**> O_matrix
) {
    // non_zero_terms_part_1 = -N_tilde + [C11] * (x_0_prime_tilde + u_prime_tilde)
    auto non_zero_terms_part_1 =
        Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>(
            "non_zero_terms_part_1"
        );
    KokkosBlas::gemm("N", "N", 1., C11, values, 0., non_zero_terms_part_1);
    KokkosBlas::axpy(-1., N_tilde, non_zero_terms_part_1);

    // non_zero_terms_part_2 = -M_tilde + [C21] * (x_0_prime_tilde  + u_prime_tilde)
    auto non_zero_terms_part_2 =
        Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>(
            "non_zero_terms_part_2"
        );
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
    const Kokkos::View<double**> N_tilde, const Kokkos::View<double**> C11,
    const Kokkos::View<double**> C12, const Kokkos::View<double**> values,
    Kokkos::View<double**> P_matrix
) {
    // non_zero_terms_part_3 = (x_0_prime_tilde + u_prime_tilde)^T * [C11]
    auto non_zero_terms_part_3 =
        Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>(
            "non_zero_terms_part_3"
        );
    KokkosBlas::gemm("T", "N", 1., values, C11, 0., non_zero_terms_part_3);
    KokkosBlas::axpy(1., N_tilde, non_zero_terms_part_3);

    // non_zero_terms_part_4 = (x_0_prime_tilde  + u_prime_tilde)^T * [C12]
    auto non_zero_terms_part_4 =
        Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>(
            "non_zero_terms_part_4"
        );
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
    const Kokkos::View<double**> N_tilde, const Kokkos::View<double**> C11,
    const Kokkos::View<double**> values, Kokkos::View<double**> Q_matrix
) {
    // Assemble the Q matrix
    // [Q]_6x6 = [
    //     [0]_3x3                                          [0]_3x3
    //     [0]_3x3      (x_0_prime_tilde + u_prime_tilde)^T * (-N_tilde + [C11] * (x_0_prime_tilde  +
    //     u_prime_tilde))
    // ]
    auto val = Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>("val");
    KokkosBlas::gemm("N", "N", 1., C11, values, 0., val);
    KokkosBlas::axpy(-1., N_tilde, val);

    Kokkos::deep_copy(Q_matrix, 0.);
    auto non_zero_term = Kokkos::subview(Q_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    KokkosBlas::gemm("T", "N", 1., values, val, 0., non_zero_term);
}

void CalculateIterationMatrixComponents(
    const Kokkos::View<double*> elastic_force_fc, const Kokkos::View<double*> pos_vector_derivatives,
    const Kokkos::View<double*> gen_coords_derivatives,
    const Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> O_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> P_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> Q_matrix
) {
    auto x0_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(pos_vector_derivatives, Kokkos::make_pair(0, 3))
    );
    auto u_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(gen_coords_derivatives, Kokkos::make_pair(0, 3))
    );

    auto values =
        Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>("values");
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

void CalculateStaticIterationMatrix(
    const Kokkos::View<double**> position_vectors, const Kokkos::View<double**> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature,
    Kokkos::View<double**> iteration_matrix
) {
    const auto n_nodes = gen_coords.extent(0);
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto nodes = Kokkos::subview(position_vectors, Kokkos::ALL, Kokkos::make_pair(0, 3));

    // Allocate Views for some required intermediate variables
    auto gen_coords_qp = Kokkos::View<double[kNumberOfLieAlgebraComponents]>("gen_coords_qp");
    auto gen_coords_derivatives_qp =
        Kokkos::View<double[kNumberOfLieAlgebraComponents]>("gen_coords_derivatives_qp");
    auto position_vector_qp =
        Kokkos::View<double[kNumberOfLieAlgebraComponents]>("position_vector_qp");
    auto pos_vector_derivatives_qp =
        Kokkos::View<double[kNumberOfLieAlgebraComponents]>("pos_vector_derivatives_qp");
    auto curvature = Kokkos::View<double[kNumberOfVectorComponents]>("curvature");
    auto sectional_strain = Kokkos::View<double[kNumberOfLieGroupComponents]>("sectional_strain");
    auto sectional_stiffness =
        Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>(
            "sectional_stiffness"
        );
    auto O_matrix =
        Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>("O_matrix");
    auto P_matrix =
        Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>("P_matrix");
    auto Q_matrix =
        Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>("Q_matrix");

    Kokkos::deep_copy(iteration_matrix, 0.);
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
                CalculateCurvature(gen_coords_qp, gen_coords_derivatives_qp, curvature);
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
                CalculateSectionalStiffness(stiffness, rotation_0, rotation, sectional_stiffness);

                // Calculate elastic forces i.e. F^C and F^D vectors
                auto elastic_forces_fc =
                    Kokkos::View<double[kNumberOfLieGroupComponents]>("elastic_forces_fc");
                auto elastic_forces_fd =
                    Kokkos::View<double[kNumberOfLieGroupComponents]>("elastic_forces_fd");
                CalculateElasticForces(
                    sectional_strain, rotation, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                    sectional_stiffness, elastic_forces_fc, elastic_forces_fd
                );

                // Calculate the iteration matrix components, i.e. O, P, and Q matrices
                CalculateIterationMatrixComponents(
                    elastic_forces_fc, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                    sectional_stiffness, O_matrix, P_matrix, Q_matrix
                );

                const auto q_weight = quadrature.GetQuadratureWeights()[k];
                Kokkos::parallel_for(
                    Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
                        {0, 0}, {kNumberOfLieGroupComponents, kNumberOfLieGroupComponents}
                    ),
                    KOKKOS_LAMBDA(const size_t ii, const size_t jj) {
                        iteration_matrix(
                            i * kNumberOfLieGroupComponents + ii,
                            j * kNumberOfLieGroupComponents + jj
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

void ConstraintsResidualVector(
    const Kokkos::View<double**> gen_coords, const Kokkos::View<double**> position_vector,
    Kokkos::View<double*> constraints_residual
) {
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
            // TODO: update when position & rotation are prescribed
            constraints_residual(0) = gen_coords(0, 0);
            constraints_residual(1) = gen_coords(0, 1);
            constraints_residual(2) = gen_coords(0, 2);
            constraints_residual(3) = rotation_vector.GetXComponent();
            constraints_residual(4) = rotation_vector.GetYComponent();
            constraints_residual(5) = rotation_vector.GetZComponent();
        }
    );
}

void ConstraintsGradientMatrix(
    const Kokkos::View<double**> gen_coords, const Kokkos::View<double**> position_vector,
    Kokkos::View<double**> constraints_gradient_matrix
) {
    auto translation_0 = Kokkos::subview(gen_coords, 0, Kokkos::make_pair(0, 3));
    auto rotation_0 = Kokkos::subview(gen_coords, 0, Kokkos::make_pair(3, 7));
    auto rotation_matrix_0 = gen_alpha_solver::EulerParameterToRotationMatrix(rotation_0);
    auto position_0 = Kokkos::subview(position_vector, 0, Kokkos::make_pair(0, 3));

    // position_cross_prod_matrix = ~{position_0} + ~{translation_0}
    auto position_0_cross_prod_matrix = gen_alpha_solver::create_cross_product_matrix(position_0);
    auto translation_0_cross_prod_matrix =
        gen_alpha_solver::create_cross_product_matrix(translation_0);
    auto position_cross_prod_matrix =
        Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]>(
            "position_cross_prod_matrix"
        );

    Kokkos::deep_copy(position_cross_prod_matrix, position_0_cross_prod_matrix);
    KokkosBlas::axpy(1., translation_0_cross_prod_matrix, position_cross_prod_matrix);

    // Assemble the constraint gradient matrix i.e. B matrix
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
