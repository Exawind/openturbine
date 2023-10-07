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

Kokkos::View<double*> Interpolate(
    Kokkos::View<double*> nodal_values, Kokkos::View<double*> interpolation_function
) {
    const auto n_nodes = nodal_values.extent(0) / kNumberOfLieAlgebraComponents;
    auto interpolated_values = Kokkos::View<double*>("interpolated_values", 7);
    Kokkos::deep_copy(interpolated_values, 0.);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {kNumberOfLieAlgebraComponents, n_nodes}
        ),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            interpolated_values(i) += interpolation_function(j) * nodal_values(j * 7 + i);
        }
    );
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
    const StiffnessMatrix& stiffness, gen_alpha_solver::RotationMatrix rotation_0,
    gen_alpha_solver::RotationMatrix rotation
) {
    auto total_rotation = gen_alpha_solver::multiply_matrix_with_matrix(
        rotation.GetRotationMatrix(), rotation_0.GetRotationMatrix()
    );

    // rotation_matrix__6x6 = [total_rotation [0]_3x3; [0]_3x3 total_rotation]
    auto rotation_matrix = Kokkos::View<double**>("rotation_matrix", 6, 6);
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
    const Kokkos::View<double*> sectional_strain, gen_alpha_solver::RotationMatrix rotation,
    const Kokkos::View<double*> position_vector_derivatives,
    const Kokkos::View<double*> gen_coords_derivatives,
    const Kokkos::View<double**> sectional_stiffness
) {
    // Calculate first part of the elastic forces i.e. F^C vector
    auto sectional_strain_next = Kokkos::View<double*>("sectional_strain_next", 6);
    Kokkos::deep_copy(sectional_strain_next, sectional_strain);
    auto R_x0 = gen_alpha_solver::multiply_matrix_with_vector(
        rotation.GetRotationMatrix(),
        gen_alpha_solver::create_vector(
            {position_vector_derivatives(0), position_vector_derivatives(1),
             position_vector_derivatives(2)}
        )
    );
    Kokkos::parallel_for(
        3, KOKKOS_LAMBDA(const size_t k) { sectional_strain_next(k) -= R_x0(k); }
    );

    auto elastic_force_fc =
        gen_alpha_solver::multiply_matrix_with_vector(sectional_stiffness, sectional_strain_next);

    // Calculate second part of the elastic forces i.e. F^D vector
    auto x0_prime_tilde =
        gen_alpha_solver::create_cross_product_matrix(gen_alpha_solver::create_vector(
            {position_vector_derivatives(0), position_vector_derivatives(1),
             position_vector_derivatives(2)}
        ));
    auto u_prime_tilde =
        gen_alpha_solver::create_cross_product_matrix(gen_alpha_solver::create_vector(
            {gen_coords_derivatives(0), gen_coords_derivatives(1), gen_coords_derivatives(2)}
        ));
    auto fd_values = gen_alpha_solver::transpose_matrix(
        gen_alpha_solver::add_matrix_with_matrix(x0_prime_tilde, u_prime_tilde)
    );

    auto elastic_force_fd = Kokkos::View<double*>("elastic_force_fd", 6);
    Kokkos::deep_copy(elastic_force_fd, 0.);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {kNumberOfVectorComponents, kNumberOfVectorComponents}
        ),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            elastic_force_fd(i + 3) += fd_values(i, j) * elastic_force_fc(j);
        }
    );

    auto elastic_forces = Kokkos::View<double*>("elastic_forces", 12);
    Kokkos::parallel_for(
        6,
        KOKKOS_LAMBDA(const size_t k) {
            elastic_forces(k) = elastic_force_fc(k);
            elastic_forces(k + 6) = elastic_force_fd(k);
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

    auto residual = Kokkos::View<double* [6]>("static_residual");
    Kokkos::deep_copy(residual, 0.);
    for (size_t i = 0; i < n_nodes; ++i) {
        for (size_t j = 0; j < n_quad_pts; ++j) {
            // Calculate several required interpolated values at the quadrature point
            const auto qp = quadrature.GetQuadraturePoints()[j];
            const auto qw = quadrature.GetQuadratureWeights()[j];
            auto shape_function = gen_alpha_solver::create_vector(LagrangePolynomial(order, qp));
            auto shape_function_derivative =
                gen_alpha_solver::create_vector(LagrangePolynomialDerivative(order, qp));

            auto gen_coords_qp = Interpolate(gen_coords, shape_function);
            auto gen_coords_derivatives_qp = Interpolate(gen_coords, shape_function_derivative);
            auto position_vector_qp = Interpolate(position_vectors, shape_function);
            auto position_vector_derivatives_qp =
                Interpolate(position_vectors, shape_function_derivative);

            // Calculate the curvature at the quadrature point
            auto curvature = CalculateCurvature(gen_coords_qp, gen_coords_derivatives_qp);

            // Calculate the sectional strain at the quadrature point based on Eq. (35)
            // in the SO(3)-based GEBT Beam document in theory guide
            auto sectional_strain = Kokkos::View<double*>("sectional_strain", 6);
            Kokkos::parallel_for(
                3,
                KOKKOS_LAMBDA(const size_t k) {
                    sectional_strain(k) =
                        position_vector_derivatives_qp(k) + gen_coords_derivatives_qp(k);
                    sectional_strain(k + 3) = curvature(k);
                }
            );

            // Calculate the sectional stiffness matrix in inertial basis
            auto q = gen_alpha_solver::Quaternion(
                gen_coords_qp(3), gen_coords_qp(4), gen_coords_qp(5), gen_coords_qp(6)
            );
            auto rotation_0 =
                gen_alpha_solver::quaternion_to_rotation_matrix(gen_alpha_solver::Quaternion(
                    position_vector_qp(3), position_vector_qp(4), position_vector_qp(5),
                    position_vector_qp(6)
                ));
            auto rotation = gen_alpha_solver::quaternion_to_rotation_matrix(q);

            auto sectional_stiffness = CalculateSectionalStiffness(stiffness, rotation_0, rotation);

            // Calculate elastic forces i.e. F^C and F^D vectors at the quadrature point
            auto elastic_forces = CalculateElasticForces(
                sectional_strain, rotation, position_vector_derivatives_qp,
                gen_coords_derivatives_qp, sectional_stiffness
            );
            auto elastic_force_fc = Kokkos::View<double*>("elastic_force_fc", 6);
            auto elastic_force_fd = Kokkos::View<double*>("elastic_force_fd", 6);
            Kokkos::parallel_for(
                6,
                KOKKOS_LAMBDA(const size_t k) {
                    elastic_force_fc(k) = elastic_forces(k);
                    elastic_force_fd(k) = elastic_forces(k + 6);
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
