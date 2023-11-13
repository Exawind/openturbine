#include "src/gebt_poc/solver.h"

#include <KokkosBlas.hpp>

#include "src/gebt_poc/element.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

UserDefinedQuadrature::UserDefinedQuadrature(
    std::vector<double> quadrature_points, std::vector<double> quadrature_weights
)
    : quadrature_points_(std::move(quadrature_points)),
      quadrature_weights_(std::move(quadrature_weights)) {
}

Kokkos::View<double*> Interpolate(
    Kokkos::View<double*> nodal_values, Kokkos::View<double*> interpolation_function, double jacobian
) {
    const auto n_nodes = nodal_values.extent(0) / kNumberOfLieAlgebraComponents;
    auto interpolated_values =
        Kokkos::View<double*>("interpolated_values", kNumberOfLieAlgebraComponents);
    Kokkos::deep_copy(interpolated_values, 0.);
    for (std::size_t i = 0; i < kNumberOfLieAlgebraComponents; ++i) {
        Kokkos::parallel_reduce(
            n_nodes,
            KOKKOS_LAMBDA(const size_t j, double& value) {
                value += interpolation_function(j) *
                         nodal_values(j * kNumberOfLieAlgebraComponents + i) / jacobian;
            },
            interpolated_values(i)
        );
    }
    return interpolated_values;
}

Kokkos::View<double*> CalculateCurvature(
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> gen_coords_derivative
) {
    auto q =
        gen_alpha_solver::Quaternion(gen_coords(3), gen_coords(4), gen_coords(5), gen_coords(6));
    auto b_matrix = gen_alpha_solver::BMatrixForQuaternions(q);

    auto q_prime = gen_alpha_solver::create_vector(
        {gen_coords_derivative(3), gen_coords_derivative(4), gen_coords_derivative(5),
         gen_coords_derivative(6)}
    );

    auto curvature = gen_alpha_solver::multiply_vector_with_scalar(
        gen_alpha_solver::multiply_matrix_with_vector(b_matrix, q_prime), 2.
    );

    return curvature;
}

void CalculateSectionalStiffness(
    const StiffnessMatrix& stiffness, Kokkos::View<double**> rotation_0,
    Kokkos::View<double**> rotation, Kokkos::View<double**> sectional_stiffness
) {
    auto total_rotation = Kokkos::View<double**>(
        "total_rotation", kNumberOfVectorComponents, kNumberOfVectorComponents
    );
    KokkosBlas::gemm("N", "N", 1., rotation, rotation_0, 0., total_rotation);

    // rotation_matrix_6x6 = [
    //    [total_rotation]          [0]_3x3
    //        [0]_3x3           total_rotation
    // ]
    auto rotation_matrix = Kokkos::View<double**>(
        "rotation_matrix", kNumberOfLieGroupComponents, kNumberOfLieGroupComponents
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
        Kokkos::View<double**>("temp", kNumberOfLieGroupComponents, kNumberOfLieGroupComponents);
    KokkosBlas::gemm(
        "N", "N", 1., rotation_matrix, stiffness.GetStiffnessMatrix(), 0., stiffness_matrix_left_rot
    );
    KokkosBlas::gemm(
        "N", "T", 1., stiffness_matrix_left_rot, rotation_matrix, 0., sectional_stiffness
    );
}

Kokkos::View<double*> CalculateElasticForces(
    const Kokkos::View<double*> sectional_strain, Kokkos::View<double**> rotation,
    const Kokkos::View<double*> pos_vector_derivatives,
    const Kokkos::View<double*> gen_coords_derivatives,
    const Kokkos::View<double**> sectional_stiffness
) {
    // Calculate first part of the elastic forces i.e. F^C vector
    auto sectional_strain_next =
        Kokkos::View<double*>("sectional_strain_next", kNumberOfLieGroupComponents);
    Kokkos::deep_copy(sectional_strain_next, sectional_strain);
    auto x0_prime = gen_alpha_solver::create_vector(
        {pos_vector_derivatives(0), pos_vector_derivatives(1), pos_vector_derivatives(2)}
    );
    auto R_x0_prime = gen_alpha_solver::multiply_matrix_with_vector(rotation, x0_prime);
    Kokkos::parallel_for(
        3, KOKKOS_LAMBDA(const size_t k) { sectional_strain_next(k) -= R_x0_prime(k); }
    );

    auto elastic_force_fc =
        gen_alpha_solver::multiply_matrix_with_vector(sectional_stiffness, sectional_strain_next);

    // Calculate second part of the elastic forces i.e. F^D vector
    auto x0_prime_tilde = gen_alpha_solver::create_cross_product_matrix(x0_prime);
    auto u_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        // create a Vector from the generalized coordinates derivatives
        gen_alpha_solver::create_vector(
            {gen_coords_derivatives(0), gen_coords_derivatives(1), gen_coords_derivatives(2)}
        )
    );
    auto fd_values = gen_alpha_solver::transpose_matrix(
        gen_alpha_solver::add_matrix_with_matrix(x0_prime_tilde, u_prime_tilde)
    );

    auto elastic_force_fd = Kokkos::View<double*>("elastic_force_fd", kNumberOfLieGroupComponents);
    Kokkos::deep_copy(elastic_force_fd, 0.);
    Kokkos::parallel_for(
        kNumberOfVectorComponents,
        KOKKOS_LAMBDA(const size_t i) {
            elastic_force_fd(i + 3) += fd_values(i, 0) * elastic_force_fc(0) +
                                       fd_values(i, 1) * elastic_force_fc(1) +
                                       fd_values(i, 2) * elastic_force_fc(2);
        }
    );

    auto elastic_forces = Kokkos::View<double*>("elastic_forces", 2 * kNumberOfLieGroupComponents);
    Kokkos::parallel_for(
        kNumberOfLieGroupComponents,
        KOKKOS_LAMBDA(const size_t k) {
            elastic_forces(k) = elastic_force_fc(k);
            elastic_forces(k + kNumberOfLieGroupComponents) = elastic_force_fd(k);
        }
    );
    return elastic_forces;
}

Kokkos::View<double*> CalculateStaticResidual(
    const Kokkos::View<double*> position_vectors, const Kokkos::View<double*> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature
) {
    const auto n_nodes = gen_coords.extent(0) / kNumberOfLieAlgebraComponents;
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    std::vector<Point> nodes;
    for (size_t i = 0; i < n_nodes; ++i) {
        nodes.emplace_back(
            position_vectors(i * kNumberOfLieAlgebraComponents),
            position_vectors(i * kNumberOfLieAlgebraComponents + 1),
            position_vectors(i * kNumberOfLieAlgebraComponents + 2)
        );
    }

    auto residual = Kokkos::View<double*>("static_residual", n_nodes * kNumberOfLieGroupComponents);
    Kokkos::deep_copy(residual, 0.);
    for (size_t i = 0; i < n_nodes; ++i) {
        const auto node_count = i;
        for (size_t j = 0; j < n_quad_pts; ++j) {
            // Calculate required interpolated values at the quadrature point
            const auto q_pt = quadrature.GetQuadraturePoints()[j];
            auto shape_function = gen_alpha_solver::create_vector(LagrangePolynomial(order, q_pt));
            auto shape_function_derivative =
                gen_alpha_solver::create_vector(LagrangePolynomialDerivative(order, q_pt));

            auto jacobian = CalculateJacobian(nodes, LagrangePolynomialDerivative(order, q_pt));
            auto gen_coords_qp = Interpolate(gen_coords, shape_function);
            auto gen_coords_derivatives_qp =
                Interpolate(gen_coords, shape_function_derivative, jacobian);
            auto position_vector_qp = Interpolate(position_vectors, shape_function);
            auto pos_vector_derivatives_qp =
                Interpolate(position_vectors, shape_function_derivative, jacobian);

            // Calculate the curvature at the quadrature point
            auto curvature = CalculateCurvature(gen_coords_qp, gen_coords_derivatives_qp);

            // Calculate the sectional strain at the quadrature point based on Eq. (35)
            // in the "SO(3)-based GEBT Beam" document in theory guide
            auto sectional_strain =
                Kokkos::View<double*>("sectional_strain", kNumberOfLieGroupComponents);
            Kokkos::parallel_for(
                kNumberOfVectorComponents,
                KOKKOS_LAMBDA(const size_t k) {
                    sectional_strain(k) =
                        pos_vector_derivatives_qp(k) + gen_coords_derivatives_qp(k);
                    sectional_strain(k + kNumberOfVectorComponents) = curvature(k);
                }
            );

            // Calculate the sectional stiffness matrix in inertial basis
            auto rotation_0 =
                gen_alpha_solver::EulerParameterToRotationMatrix(gen_alpha_solver::create_vector(
                    {position_vector_qp(3), position_vector_qp(4), position_vector_qp(5),
                     position_vector_qp(6)}
                ));
            auto rotation =
                gen_alpha_solver::EulerParameterToRotationMatrix(gen_alpha_solver::create_vector(
                    {gen_coords_qp(3), gen_coords_qp(4), gen_coords_qp(5), gen_coords_qp(6)}
                ));

            auto sectional_stiffness = Kokkos::View<double**>(
                "sectional_stiffness", kNumberOfLieGroupComponents, kNumberOfLieGroupComponents
            );
            CalculateSectionalStiffness(stiffness, rotation_0, rotation, sectional_stiffness);

            // Calculate elastic forces i.e. F^C and F^D vectors at the quadrature point
            auto elastic_forces = CalculateElasticForces(
                sectional_strain, rotation, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                sectional_stiffness
            );
            auto elastic_force_fc =
                Kokkos::View<double*>("elastic_force_fc", kNumberOfLieGroupComponents);
            auto elastic_force_fd =
                Kokkos::View<double*>("elastic_force_fd", kNumberOfLieGroupComponents);
            Kokkos::parallel_for(
                kNumberOfLieGroupComponents,
                KOKKOS_LAMBDA(const size_t k) {
                    elastic_force_fc(k) = elastic_forces(k);
                    elastic_force_fd(k) = elastic_forces(k + kNumberOfLieGroupComponents);
                }
            );

            // Calculate the residual at the quadrature point
            const auto q_weight = quadrature.GetQuadratureWeights()[j];
            Kokkos::parallel_for(
                kNumberOfLieGroupComponents,
                KOKKOS_LAMBDA(const size_t i) {
                    residual(node_count * kNumberOfLieGroupComponents + i) +=
                        q_weight * (shape_function_derivative(node_count) * elastic_force_fc(i) +
                                    jacobian * shape_function(node_count) * elastic_force_fd(i));
                }
            );
        }
    }
    return residual;
}

void CalculateOMatrix(
    const Kokkos::View<double**> N_tilde, const Kokkos::View<double**> M_tilde,
    const Kokkos::View<double**> C11, const Kokkos::View<double**> C21,
    const Kokkos::View<double**> values, Kokkos::View<double**> O_matrix
) {
    // non_zero_terms_part_1 = -N_tilde + [C11] * (x_0_prime_tilde + u_prime_tilde)
    auto non_zero_terms_part_1 = Kokkos::View<double**>(
        "non_zero_terms_part_1", kNumberOfVectorComponents, kNumberOfVectorComponents
    );
    KokkosBlas::gemm("N", "N", 1., C11, values, 0., non_zero_terms_part_1);
    KokkosBlas::axpy(-1., N_tilde, non_zero_terms_part_1);

    // non_zero_terms_part_2 = -M_tilde + [C21] * (x_0_prime_tilde  + u_prime_tilde)
    auto non_zero_terms_part_2 = Kokkos::View<double**>(
        "non_zero_terms_part_2", kNumberOfVectorComponents, kNumberOfVectorComponents
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
    auto non_zero_terms_part_3 = Kokkos::View<double**>(
        "non_zero_terms_part_3", kNumberOfVectorComponents, kNumberOfVectorComponents
    );
    KokkosBlas::gemm("T", "N", 1., values, C11, 0., non_zero_terms_part_3);
    KokkosBlas::axpy(1., N_tilde, non_zero_terms_part_3);

    // non_zero_terms_part_4 = (x_0_prime_tilde  + u_prime_tilde)^T * [C12]
    auto non_zero_terms_part_4 = Kokkos::View<double**>(
        "non_zero_terms_part_4", kNumberOfVectorComponents, kNumberOfVectorComponents
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
    auto val = Kokkos::View<double**>("val", kNumberOfVectorComponents, kNumberOfVectorComponents);
    KokkosBlas::gemm("N", "N", 1., C11, values, 0., val);
    KokkosBlas::axpy(-1., N_tilde, val);

    Kokkos::deep_copy(Q_matrix, 0.);
    auto non_zero_term = Kokkos::subview(Q_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    KokkosBlas::gemm("T", "N", 1., values, val, 0., non_zero_term);
}

void CalculateIterationMatrixComponents(
    const Kokkos::View<double*> elastic_force_fc, const Kokkos::View<double*> pos_vector_derivatives,
    const Kokkos::View<double*> gen_coords_derivatives,
    const Kokkos::View<double**> sectional_stiffness, Kokkos::View<double**> O_P_Q_matrices
) {
    auto x0_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(pos_vector_derivatives, Kokkos::make_pair(0, 3))
    );
    auto u_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(gen_coords_derivatives, Kokkos::make_pair(0, 3))
    );

    auto values =
        Kokkos::View<double**>("values", kNumberOfVectorComponents, kNumberOfVectorComponents);
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
    auto O_matrix =
        Kokkos::subview(O_P_Q_matrices, Kokkos::make_pair(0, 6), Kokkos::make_pair(0, 6));
    CalculateOMatrix(N_tilde, M_tilde, C11, C21, values, O_matrix);

    auto P_matrix =
        Kokkos::subview(O_P_Q_matrices, Kokkos::make_pair(6, 12), Kokkos::make_pair(0, 6));
    CalculatePMatrix(N_tilde, C11, C12, values, P_matrix);

    auto Q_matrix =
        Kokkos::subview(O_P_Q_matrices, Kokkos::make_pair(12, 18), Kokkos::make_pair(0, 6));
    CalculateQMatrix(N_tilde, C11, values, Q_matrix);
}

void CalculateStaticIterationMatrix(
    const Kokkos::View<double*> position_vectors, const Kokkos::View<double*> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature,
    Kokkos::View<double**> iteration_matrix
) {
    const auto n_nodes = gen_coords.extent(0) / kNumberOfLieAlgebraComponents;
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    std::vector<Point> nodes;
    for (size_t i = 0; i < n_nodes; ++i) {
        nodes.emplace_back(
            position_vectors(i * kNumberOfLieAlgebraComponents),
            position_vectors(i * kNumberOfLieAlgebraComponents + 1),
            position_vectors(i * kNumberOfLieAlgebraComponents + 2)
        );
    }

    Kokkos::deep_copy(iteration_matrix, 0.);
    for (size_t i = 0; i < n_nodes; ++i) {
        for (size_t j = 0; j < n_nodes; ++j) {
            for (size_t k = 0; k < n_quad_pts; ++k) {
                // Calculate required interpolated values at the quadrature point
                const auto q_pt = quadrature.GetQuadraturePoints()[k];
                auto shape_function =
                    gen_alpha_solver::create_vector(LagrangePolynomial(order, q_pt));
                auto shape_function_derivative =
                    gen_alpha_solver::create_vector(LagrangePolynomialDerivative(order, q_pt));

                auto jacobian = CalculateJacobian(nodes, LagrangePolynomialDerivative(order, q_pt));
                auto gen_coords_qp = Interpolate(gen_coords, shape_function);
                auto gen_coords_derivatives_qp =
                    Interpolate(gen_coords, shape_function_derivative, jacobian);
                auto position_vector_qp = Interpolate(position_vectors, shape_function);
                auto pos_vector_derivatives_qp =
                    Interpolate(position_vectors, shape_function_derivative, jacobian);

                // Calculate the curvature at the quadrature point
                auto curvature = CalculateCurvature(gen_coords_qp, gen_coords_derivatives_qp);

                // Calculate the sectional strain at the quadrature point based on Eq. (35)
                // in the "SO(3)-based GEBT Beam" document in theory guide
                auto sectional_strain =
                    Kokkos::View<double*>("sectional_strain", kNumberOfLieGroupComponents);
                Kokkos::parallel_for(
                    kNumberOfVectorComponents,
                    KOKKOS_LAMBDA(const size_t k) {
                        sectional_strain(k) =
                            pos_vector_derivatives_qp(k) + gen_coords_derivatives_qp(k);
                        sectional_strain(k + kNumberOfVectorComponents) = curvature(k);
                    }
                );

                // Calculate the sectional stiffness matrix in inertial basis
                auto rotation_0 = gen_alpha_solver::EulerParameterToRotationMatrix(
                    Kokkos::subview(position_vector_qp, Kokkos::make_pair(3, 7))
                );
                auto rotation = gen_alpha_solver::EulerParameterToRotationMatrix(
                    Kokkos::subview(gen_coords_qp, Kokkos::make_pair(3, 7))
                );

                auto sectional_stiffness = Kokkos::View<double**>(
                    "sectional_stiffness", kNumberOfLieGroupComponents, kNumberOfLieGroupComponents
                );
                CalculateSectionalStiffness(stiffness, rotation_0, rotation, sectional_stiffness);

                // Calculate elastic forces i.e. F^C and F^D vectors at the quadrature point
                auto elastic_forces = CalculateElasticForces(
                    sectional_strain, rotation, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                    sectional_stiffness
                );
                auto elastic_force_fc =
                    Kokkos::View<double*>("elastic_force_fc", kNumberOfLieGroupComponents);
                Kokkos::parallel_for(
                    kNumberOfLieGroupComponents,
                    KOKKOS_LAMBDA(const size_t k) { elastic_force_fc(k) = elastic_forces(k); }
                );

                // Calculate the iteration matrix components at the quadrature point
                auto iteration_matrix_components = Kokkos::View<double**>(
                    "O_P_Q_matrices", 3 * kNumberOfLieGroupComponents, kNumberOfLieGroupComponents
                );
                CalculateIterationMatrixComponents(
                    elastic_force_fc, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                    sectional_stiffness, iteration_matrix_components
                );
                auto O_matrix = Kokkos::subview(
                    iteration_matrix_components, Kokkos::make_pair(0, 6), Kokkos::make_pair(0, 6)
                );
                auto P_matrix = Kokkos::subview(
                    iteration_matrix_components, Kokkos::make_pair(6, 12), Kokkos::make_pair(0, 6)
                );
                auto Q_matrix = Kokkos::subview(
                    iteration_matrix_components, Kokkos::make_pair(12, 18), Kokkos::make_pair(0, 6)
                );

                // Calculate the iteration matrix at the quadrature point
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
                             (shape_function(i) * P_matrix(ii, jj) * shape_function_derivative(j) +
                              shape_function(i) * Q_matrix(ii, jj) * shape_function(j) * jacobian +
                              shape_function_derivative(i) * sectional_stiffness(ii, jj) *
                                  shape_function_derivative(j) / jacobian +
                              shape_function_derivative(i) * O_matrix(ii, jj) * shape_function(j));
                    }
                );
            }
        }
    }
}

void ConstraintsResidualVector(
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> position_vector,
    Kokkos::View<double*> constraint_residual
) {
    auto translation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(0, 3));
    auto rotation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(3, 7));
    auto rotation_matrix_0 = gen_alpha_solver::EulerParameterToRotationMatrix(rotation_0);
    auto position_0 = Kokkos::subview(position_vector, Kokkos::make_pair(0, 3));

    // position = position_0 + translation_0
    auto position = Kokkos::View<double*>("position", kNumberOfVectorComponents);
    Kokkos::deep_copy(position, position_0);
    KokkosBlas::axpy(1., translation_0, position);

    // rotated_position = rotation_matrix_0 * position
    auto rotated_position = Kokkos::View<double*>("rotated_position", kNumberOfVectorComponents);
    KokkosBlas::gemv("N", 1., rotation_matrix_0, position, 0., rotated_position);

    // Assemble the constraint residual vector
    // {constraint_residual}_6x1 = {
    //    {translation_0}_3x1
    //    {rotated_position - position_0}_3x1
    // }
    auto constraint_residual_1 = Kokkos::subview(constraint_residual, Kokkos::make_pair(0, 3));
    auto constraint_residual_2 = Kokkos::subview(constraint_residual, Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(constraint_residual_1, translation_0);
    Kokkos::deep_copy(constraint_residual_2, rotated_position);
    KokkosBlas::axpy(-1., position_0, constraint_residual_2);
}

void ConstraintsGradientMatrix(
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> position_vector,
    Kokkos::View<double**> constraint_gradient_matrix
) {
    auto translation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(0, 3));
    auto rotation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(3, 7));
    auto rotation_matrix_0 = gen_alpha_solver::EulerParameterToRotationMatrix(rotation_0);
    auto position_0 = Kokkos::subview(position_vector, Kokkos::make_pair(0, 3));

    // position_cross_prod_matrix = ~{position_0} + ~{translation_0}
    auto position_0_cross_prod_matrix = gen_alpha_solver::create_cross_product_matrix(position_0);
    auto translation_0_cross_prod_matrix =
        gen_alpha_solver::create_cross_product_matrix(translation_0);
    auto position_cross_prod_matrix = Kokkos::View<double**>("position_cross_prod_matrix", 3, 3);
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
    Kokkos::deep_copy(constraint_gradient_matrix, 0.);
    auto B11 = Kokkos::subview(
        constraint_gradient_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3)
    );
    auto B21 = Kokkos::subview(
        constraint_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3)
    );
    auto B22 = Kokkos::subview(
        constraint_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6)
    );

    auto hostB11 = Kokkos::create_mirror_view(B11);
    hostB11(0, 0) = 1.0;
    hostB11(1, 1) = 1.0;
    hostB11(2, 2) = 1.0;
    Kokkos::deep_copy(B11, hostB11);

    KokkosBlas::scal(B21, -1.0, rotation_matrix_0);
    KokkosBlas::gemm("N", "N", -1.0, rotation_matrix_0, position_cross_prod_matrix, 0.0, B22);
}

}  // namespace openturbine::gebt_poc
