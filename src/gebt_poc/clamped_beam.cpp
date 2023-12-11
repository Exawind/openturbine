#include "src/gebt_poc/clamped_beam.h"

#include <KokkosBlas.hpp>

#include "static_beam_element.h"

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

void BMatrix(Kokkos::View<double**> constraints_gradient_matrix) {
    // Assemble the constraint gradient matrix i.e. B matrix
    // [B]_6x(n+1) = [
    //     [I]_3x3        [0]       [0]   ....  [0]
    //        [0]       [I]_3x3     [0]   ....  [0]
    // ]
    // where
    // [I]_3x3 = [1]_3x3
    // [0] = [0]_3x3

    Kokkos::deep_copy(constraints_gradient_matrix, 0.);
    auto B11 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3)
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
            B22(0, 0) = 1.;
            B22(1, 1) = 1.;
            B22(2, 2) = 1.;
        }
    );
}

ClampedBeamLinearizationParameters::ClampedBeamLinearizationParameters(
    Kokkos::View<double*> position_vectors, StiffnessMatrix stiffness_matrix,
    UserDefinedQuadrature quadrature
)
    : position_vectors_(position_vectors),
      stiffness_matrix_(stiffness_matrix),
      quadrature_(quadrature) {
}

void ClampedBeamLinearizationParameters::ResidualVector(
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double*> residual
) {
    // The residual vector for the generalized coordinates is given by
    // {residual} = {
    //     {residual_gen_coords} + {constraints_part2},
    //     {residual_constraints}
    // }
    const size_t zero{0};
    const auto size_dofs = velocity.extent(0) * velocity.extent(1);
    const auto size_constraints = lagrange_multipliers.extent(0);
    const auto size_residual = size_dofs + size_constraints;

    // Part 1: Calculate the residual vector for the generalized coordinates
    auto gen_coords_1D =
        Kokkos::View<double*>("gen_coords_1D", gen_coords.extent(0) * gen_coords.extent(1));
    Convert2DViewTo1DView(gen_coords, gen_coords_1D);

    Kokkos::deep_copy(residual, 0.0);
    auto residual_gen_coords = Kokkos::subview(residual, Kokkos::make_pair(zero, size_dofs));
    CalculateStaticResidual(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, residual_gen_coords
    );

    // Part 2: Calculate the residual vector for the constraints
    // {R_c} = {B(q)}^T * {lambda}
    auto constraints_gradient_matrix =
        Kokkos::View<double**>("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    auto constraints_part2 = Kokkos::View<double*>("constraints_part2", size_dofs);
    KokkosBlas::gemv(
        "T", 1., constraints_gradient_matrix, lagrange_multipliers, 0., constraints_part2
    );

    KokkosBlas::axpy(1., constraints_part2, residual_gen_coords);

    auto residual_constraints =
        Kokkos::subview(residual, Kokkos::make_pair(size_dofs, size_residual));
    ConstraintsResidualVector(gen_coords_1D, position_vectors_, residual_constraints);

    if constexpr (Kokkos::SpaceAccessibility<
                      Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible) {
        auto log = util::Log::Get();
        log->Debug("residual vector: \n");
        for (size_t i = 0; i < residual.extent(0); ++i) {
            log->Debug(std::to_string(residual(i)) + "\n");
        }
    }
}

void ClampedBeamLinearizationParameters::IterationMatrix(
    const double& h, [[maybe_unused]] const double& beta_prime,
    [[maybe_unused]] const double& gamma_prime,
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double**> iteration_matrix
) {
    // Iteration matrix for the static beam element is given by
    // [iteration matrix] = [
    //     [K_t(q,v,v',Lambda,t)] * [T(h dq)]                   [B(q)^T]]
    //          [ B(q) ] * [T(h dq)]                               [0]
    // ]
    // where,
    // [K_t(q,v,v',Lambda,t)] = Tangent stiffness matrix
    // [T(h dq)] = Tangent operator
    // [B(q)] = Constraint gradient matrix
    const size_t zero{0};
    const auto size_dofs = velocity.extent(0) * velocity.extent(1);
    const auto size_constraints = lagrange_multipliers.extent(0);
    const auto size_iteration = size_dofs + size_constraints;
    const auto n_nodes = velocity.extent(0);

    auto gen_coords_1D =
        Kokkos::View<double*>("gen_coords_1D", gen_coords.extent(0) * gen_coords.extent(1));
    Convert2DViewTo1DView(gen_coords, gen_coords_1D);

    // Assemble the tangent operator (same size as the stiffness matrix)
    auto tangent_operator = Kokkos::View<double**>("tangent_operator", size_dofs, size_dofs);
    Kokkos::deep_copy(tangent_operator, 0.0);
    for (size_t i = 0; i < n_nodes; ++i) {
        auto delta_gen_coords_node = Kokkos::subview(delta_gen_coords, i, Kokkos::make_pair(3, 6));
        KokkosBlas::scal(delta_gen_coords_node, h, delta_gen_coords_node);
        auto tangent_operator_node = Kokkos::subview(
            tangent_operator,
            Kokkos::make_pair(
                i * kNumberOfLieAlgebraComponents, (i + 1) * kNumberOfLieAlgebraComponents
            ),
            Kokkos::make_pair(
                i * kNumberOfLieAlgebraComponents, (i + 1) * kNumberOfLieAlgebraComponents
            )
        );
        TangentOperator(delta_gen_coords_node, tangent_operator_node);
    }

    Kokkos::deep_copy(iteration_matrix, 0.0);

    // Calculate the beam element static iteration matrix
    auto iteration_matrix_local =
        Kokkos::View<double**>("iteration_matrix_local", size_dofs, size_dofs);
    CalculateStaticIterationMatrix(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, iteration_matrix_local
    );

    // Combine beam element static iteration matrix with constraints into quadrant 1
    // quadrant_1 = K_t + K_t_part2
    auto quadrant_1 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs), Kokkos::make_pair(zero, size_dofs)
    );
    KokkosBlas::gemm("N", "N", 1.0, iteration_matrix_local, tangent_operator, 0.0, quadrant_1);

    // quadrant_2 = Transpose([B(q)])
    auto quadrant_2 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs),
        Kokkos::make_pair(size_dofs, size_iteration)
    );
    auto constraints_gradient_matrix =
        Kokkos::View<double**>("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    auto temp = gen_alpha_solver::transpose_matrix(constraints_gradient_matrix);
    Kokkos::deep_copy(quadrant_2, temp);

    // quadrant_3 = B(q) * T(h dq)
    auto quadrant_3 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(size_dofs, size_iteration),
        Kokkos::make_pair(zero, size_dofs)
    );
    KokkosBlas::gemm("N", "N", 1.0, constraints_gradient_matrix, tangent_operator, 0.0, quadrant_3);

    // auto log = util::Log::Get();
    // log->Debug("iteration matrix: \n");
    // for (size_t i = 0; i < iteration_matrix.extent(0); ++i) {
    //     for (size_t j = 0; j < iteration_matrix.extent(1); ++j) {
    //         log->Debug(std::to_string(iteration_matrix(i, j)) + "\n");
    //     }
    //     log->Debug("\n");
    // }
}
}  // namespace openturbine::gebt_poc
