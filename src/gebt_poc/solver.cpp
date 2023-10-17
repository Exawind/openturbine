#include "src/gebt_poc/solver.h"

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

Kokkos::View<double**> CalculateSectionalStiffness(
    const StiffnessMatrix& stiffness, Kokkos::View<double**> rotation_0,
    Kokkos::View<double**> rotation
) {
    auto total_rotation = gen_alpha_solver::multiply_matrix_with_matrix(rotation, rotation_0);

    // rotation_matrix_6x6 = [total_rotation [0]_3x3; [0]_3x3 total_rotation]
    auto rotation_matrix = Kokkos::View<double**>(
        "rotation_matrix", kNumberOfLieGroupComponents, kNumberOfLieGroupComponents
    );
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {kNumberOfVectorComponents, kNumberOfVectorComponents}
        ),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            rotation_matrix(i, j) = total_rotation(i, j);
            rotation_matrix(i + 3, j + 3) = total_rotation(i, j);
        }
    );

    auto sectional_stiffness = gen_alpha_solver::multiply_matrix_with_matrix(
        gen_alpha_solver::multiply_matrix_with_matrix(
            rotation_matrix, stiffness.GetStiffnessMatrix()
        ),
        gen_alpha_solver::transpose_matrix(rotation_matrix)
    );

    return sectional_stiffness;
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

            auto sectional_stiffness = CalculateSectionalStiffness(stiffness, rotation_0, rotation);

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

Kokkos::View<double**> CalculateOMatrix(
    const Kokkos::View<double*> elastic_force_fc, const Kokkos::View<double*> pos_vector_derivatives,
    const Kokkos::View<double*> gen_coords_derivatives,
    const Kokkos::View<double**> sectional_stiffness
) {
    // [O]_6x6 = [
    //     [0]_3x3      -N_tilde + [C11] * (x_0_prime_tilde + u_prime_tilde)
    //     [0]_3x3      -M_tilde + [C21] * (x_0_prime_tilde  + u_prime_tilde)
    // ]

    // N consists of first three components of the fc vector
    auto N = Kokkos::View<double*>("N", kNumberOfVectorComponents);
    Kokkos::parallel_for(
        kNumberOfVectorComponents, KOKKOS_LAMBDA(const size_t i) { N(i) = elastic_force_fc(i); }
    );

    // M consists of last three components of the fc vector
    auto M = Kokkos::View<double*>("M", kNumberOfVectorComponents);
    Kokkos::parallel_for(
        kNumberOfVectorComponents,
        KOKKOS_LAMBDA(const size_t i) { M(i) = elastic_force_fc(i + kNumberOfVectorComponents); }
    );

    // C11 consists of the (0, 0) -> (2, 2) components of the sectional stiffness matrix
    auto C11 = Kokkos::View<double**>("C11", 3, 3);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {3, 3}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) { C11(i, j) = sectional_stiffness(i, j); }
    );

    // C21 consists of the (3, 0) -> (5, 2) components of the sectional stiffness matrix
    auto C21 = Kokkos::View<double**>("C21", 3, 3);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {3, 3}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            C21(i, j) = sectional_stiffness(i + kNumberOfVectorComponents, j);
        }
    );

    // Calculate the two non-zero submatrices
    auto x0_prime_tilde =
        gen_alpha_solver::create_cross_product_matrix(gen_alpha_solver::create_vector(
            {pos_vector_derivatives(0), pos_vector_derivatives(1), pos_vector_derivatives(2)}
        ));
    auto u_prime_tilde =
        gen_alpha_solver::create_cross_product_matrix(gen_alpha_solver::create_vector(
            {gen_coords_derivatives(0), gen_coords_derivatives(1), gen_coords_derivatives(2)}
        ));
    auto values = gen_alpha_solver::add_matrix_with_matrix(x0_prime_tilde, u_prime_tilde);
    auto N_tilde = gen_alpha_solver::multiply_matrix_with_scalar(
        gen_alpha_solver::create_cross_product_matrix(N), -1.
    );
    auto M_tilde = gen_alpha_solver::multiply_matrix_with_scalar(
        gen_alpha_solver::create_cross_product_matrix(M), -1.
    );

    auto non_zero_terms_part_1 = gen_alpha_solver::add_matrix_with_matrix(
        N_tilde, gen_alpha_solver::multiply_matrix_with_matrix(C11, values)
    );
    auto non_zero_terms_part_2 = gen_alpha_solver::add_matrix_with_matrix(
        M_tilde, gen_alpha_solver::multiply_matrix_with_matrix(C21, values)
    );

    // Assemble the O matrix
    auto O_matrix =
        Kokkos::View<double**>("O_matrix", kNumberOfLieGroupComponents, kNumberOfLieGroupComponents);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {kNumberOfVectorComponents, kNumberOfVectorComponents}
        ),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            O_matrix(i, j) = 0.;
            O_matrix(i + kNumberOfVectorComponents, j) = 0.;
            O_matrix(i, j + kNumberOfVectorComponents) = non_zero_terms_part_1(i, j);
            O_matrix(i + kNumberOfVectorComponents, j + kNumberOfVectorComponents) =
                non_zero_terms_part_2(i, j);
        }
    );
    return O_matrix;
}

}  // namespace openturbine::gebt_poc
