#include "src/gebt_poc/solver.h"

#include "src/gen_alpha_poc/quaternion.h"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

UserDefinedQuadrature::UserDefinedQuadrature(
    std::vector<double> quadrature_points, std::vector<double> quadrature_weights
)
    : quadrature_points_(std::move(quadrature_points)),
      quadrature_weights_(std::move(quadrature_weights)) {
}

Kokkos::View<double*> Interpolate(Kokkos::View<double*> nodal_values, double quadrature_pt) {
    const auto n_nodes = nodal_values.extent(0) / kNumberOfLieAlgebraComponents;
    auto shape_function = LagrangePolynomial(n_nodes - 1, quadrature_pt);

    auto interpolated_values = Kokkos::View<double*>("interpolated_values", 7);
    Kokkos::deep_copy(interpolated_values, 0.);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {kNumberOfLieAlgebraComponents, n_nodes}
        ),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            interpolated_values(i) += shape_function[j] * nodal_values(j * 7 + i);
        }
    );

    return interpolated_values;
}

Kokkos::View<double*> CalculateStaticResidual(
    const Kokkos::View<double*> position_vectors, const Kokkos::View<double*> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature
) {
    const auto n_nodes = gen_coords.extent(0) / kNumberOfLieAlgebraComponents;
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.GetNumberOfQuadraturePoints();

    auto residual = Kokkos::View<double* [6]>("static_residual");
    Kokkos::deep_copy(residual, 0.);
    for (size_t i = 0; i < n_nodes; ++i) {
        for (size_t j = 0; j < n_quad_pts; ++j) {
            // Calculate the interpolated values at the quadrature point of the element
            const auto qp = quadrature.GetQuadraturePoints()[j];
            const auto qw = quadrature.GetQuadratureWeights()[j];
            auto shape_function = LagrangePolynomial(order, qp);
            auto shape_function_derivative = LagrangePolynomialDerivative(order, qp);

            auto gen_coords_qp = Kokkos::View<double*>(
                "gen_coords_at_quadrature_point", kNumberOfLieAlgebraComponents
            );
            auto gen_coords_derivatives_qp = Kokkos::View<double*>(
                "gen_coords_derivatives_at_quadrature_point", kNumberOfLieAlgebraComponents
            );
            auto position_vector_qp = Kokkos::View<double*>(
                "position_vector_at_quadrature_point", kNumberOfLieAlgebraComponents
            );
            auto position_vector_derivatives_qp = Kokkos::View<double*>(
                "position_vector_derivatives_at_quadrature_point", kNumberOfLieAlgebraComponents
            );
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
                    {0, 0}, {kNumberOfLieAlgebraComponents, n_nodes}
                ),
                KOKKOS_LAMBDA(const size_t i, const size_t j) {
                    gen_coords_qp(i) += shape_function[j] * gen_coords(j * 7 + i);
                    gen_coords_derivatives_qp(i) +=
                        shape_function_derivative[j] * gen_coords(j * 7 + i);
                    position_vector_qp(i) += shape_function[j] * position_vectors(j * 7 + i);
                    position_vector_derivatives_qp(i) +=
                        shape_function_derivative[j] * position_vectors(j * 7 + i);
                }
            );

            // Calculate the curvature vector at the quadrature point
            auto q = gen_alpha_solver::Quaternion(
                gen_coords_qp(3), gen_coords_qp(4), gen_coords_qp(5), gen_coords_qp(6)
            );
            auto b_matrix_quaternion = gen_alpha_solver::BMatrixForQuaternions(q);
            auto q_prime = gen_alpha_solver::create_vector(
                {gen_coords_derivatives_qp(3), gen_coords_derivatives_qp(4),
                 gen_coords_derivatives_qp(5), gen_coords_derivatives_qp(6)}
            );
            auto curvature =
                gen_alpha_solver::multiply_matrix_with_vector(b_matrix_quaternion, q_prime);

            // Calculate the strain vector at the quadrature point
            auto strain = Kokkos::View<double*>("strain", 6);
            Kokkos::parallel_for(
                3,
                KOKKOS_LAMBDA(const size_t k) {
                    strain(k) = position_vector_derivatives_qp(k) + gen_coords_derivatives_qp(k);
                    strain(k + 3) = curvature(k);
                }
            );

            // Calculate the stiffness matrix at the quadrature point
            auto rotation_0 =
                gen_alpha_solver::quaternion_to_rotation_matrix(gen_alpha_solver::Quaternion(
                    position_vector_qp(3), position_vector_qp(4), position_vector_qp(5),
                    position_vector_qp(6)
                ));
            auto rotation = gen_alpha_solver::quaternion_to_rotation_matrix(q);
            auto total_rotation = gen_alpha_solver::multiply_matrix_with_matrix(
                rotation.GetRotationMatrix(), rotation_0.GetRotationMatrix()
            );

            auto stiffness_qp = gen_alpha_solver::multiply_matrix_with_matrix(
                gen_alpha_solver::multiply_matrix_with_matrix(
                    total_rotation, stiffness.GetStiffnessMatrix()
                ),
                gen_alpha_solver::transpose_matrix(total_rotation)
            );

            // Calculate F_c vector at the quadrature point
            auto strain_next = Kokkos::View<double*>("strain_next", 6);
            Kokkos::deep_copy(strain_next, strain);
            auto R_x0 = gen_alpha_solver::multiply_matrix_with_vector(
                rotation.GetRotationMatrix(),
                gen_alpha_solver::create_vector(
                    {position_vector_derivatives_qp(0), position_vector_derivatives_qp(1),
                     position_vector_derivatives_qp(2)}
                )
            );
            Kokkos::parallel_for(
                3, KOKKOS_LAMBDA(const size_t k) { strain_next(k) -= R_x0(k); }
            );

            auto fc = gen_alpha_solver::multiply_matrix_with_vector(stiffness_qp, strain_next);

            // Calculate F_d vector at the quadrature point
            auto x0_prime_tilde =
                gen_alpha_solver::create_cross_product_matrix(gen_alpha_solver::create_vector(
                    {position_vector_qp(0), position_vector_qp(1), position_vector_qp(2)}
                ));
            auto u_prime_tilde =
                gen_alpha_solver::create_cross_product_matrix(gen_alpha_solver::create_vector(
                    {gen_coords_derivatives_qp(0), gen_coords_derivatives_qp(1),
                     gen_coords_derivatives_qp(2)}
                ));
            auto fd_values = gen_alpha_solver::transpose_matrix(
                gen_alpha_solver::add_matrix_with_matrix(x0_prime_tilde, u_prime_tilde)
            );

            auto fd = Kokkos::View<double*>("fd", 6);
            Kokkos::deep_copy(fd, 0.);
            Kokkos::parallel_for(
                3,
                KOKKOS_LAMBDA(const size_t k) {
                    fd(k) =
                        fd_values(k, 0) * fc(0) + fd_values(k, 1) * fc(1) + fd_values(k, 2) * fc(2);
                }
            );

            // residual[[(i - 1)*6 + 1 ;; (i - 1)*6 + 6 ]] =
            // residual[[(i - 1)*6 + 1 ;; (i - 1)*6 + 6 ]] + (FC*
            //     Limit[Limit[hd[x, y], y -> \[Xi]j[[basis]]], x -> \[Xi]] +
            //     Sqrt[Limit[jacsquared[x], x -> \[Xi]]]*FD*
            //     Limit[Limit[h[x, y], y -> \[Xi]j[[basis]]], x -> \[Xi]]) \[Xi]w;

            // Calculate the residual at the quadrature point
            // auto phi_prime = gen_alpha_solver::create_vector(shape_function_derivative);
            // auto first_part = Kokkos::View<double*>("first_part", 6);
            // Kokkos::parallel_for(
            //     6, KOKKOS_LAMBDA(const size_t k
            //        ) { gen_coords_qp(k) += gen_alpha_solver::dot_product(phi_prime, fc) * qw; }
            // );

            // //  gen_coords_qp(i) += shape_function[j] * gen_coords(j * 7 + i);

            // Kokkos::parallel_for(
            //     Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
            //         {0, 0}, {kNumberOfLieGroupComponents, n_nodes}
            //     ),
            //     KOKKOS_LAMBDA(const size_t i, const size_t j) {
            //         residual(i, j) += qw * (shape_function_derivative[j] * gen_coords_qp(i));
            //     }
            // );
        }
    }
}

}  // namespace openturbine::gebt_poc
