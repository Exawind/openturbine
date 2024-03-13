#include "src/gebt_poc/static_beam_element.h"

#include <KokkosBlas.hpp>

#include "src/gebt_poc/ElementalConstraintForcesResidual.hpp"
#include "src/gebt_poc/ElementalStaticForcesResidual.hpp"
#include "src/gebt_poc/ElementalStaticStiffnessMatrix.hpp"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

void BMatrix(View2D constraints_gradient_matrix) {
    // Assemble the constraint gradient matrix i.e. B matrix
    // [B]_6x(n+1) = [
    //     [I]_3x3        [0]       [0]   ....  [0]
    //        [0]       [I]_3x3     [0]   ....  [0]
    // ]
    // where
    // [I]_3x3 = [1]_3x3
    // [0] = [0]_3x3

    Kokkos::deep_copy(constraints_gradient_matrix, 0.);
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            constraints_gradient_matrix(0, 0) = 1.;
            constraints_gradient_matrix(1, 1) = 1.;
            constraints_gradient_matrix(2, 2) = 1.;
            constraints_gradient_matrix(3, 3) = 1.;
            constraints_gradient_matrix(4, 4) = 1.;
            constraints_gradient_matrix(5, 5) = 1.;
        }
    );
}

StaticBeamLinearizationParameters::StaticBeamLinearizationParameters(
    LieGroupFieldView position_vectors, View2D_6x6 stiffness_matrix, Quadrature quadrature
)
    : position_vectors_(position_vectors),
      stiffness_matrix_(stiffness_matrix),
      quadrature_(quadrature) {
}

void StaticBeamLinearizationParameters::ResidualVector(
    LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type velocity,
    [[maybe_unused]] LieAlgebraFieldView::const_type acceleration,
    View1D::const_type lagrange_multipliers,
    [[maybe_unused]] const gen_alpha_solver::TimeStepper& time_stepper, View1D residual
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
    Kokkos::deep_copy(residual, 0.0);
    auto residual_gen_coords = Kokkos::subview(residual, Kokkos::make_pair(zero, size_dofs));
    ElementalStaticForcesResidual(
        position_vectors_, gen_coords, stiffness_matrix_, quadrature_, residual_gen_coords
    );

    // Part 2: Calculate the residual vector for the constraints
    // {R_c} = {B(q)}^T * {lambda}
    auto constraints_gradient_matrix =
        View2D("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    KokkosBlas::gemv(
        "T", 1., constraints_gradient_matrix, lagrange_multipliers, 1., residual_gen_coords
    );

    auto residual_constraints =
        Kokkos::subview(residual, Kokkos::make_pair(size_dofs, size_residual));
    ElementalConstraintForcesResidual(gen_coords, residual_constraints);
}

void StaticBeamLinearizationParameters::IterationMatrix(
    double h, [[maybe_unused]] double beta_prime, [[maybe_unused]] double gamma_prime,
    LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type delta_gen_coords,
    LieAlgebraFieldView::const_type velocity,
    [[maybe_unused]] LieAlgebraFieldView::const_type acceleration,
    View1D::const_type lagrange_multipliers, View2D iteration_matrix
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

    // Assemble the tangent operator (same size as the stiffness matrix)
    auto tangent_operator = View2D("tangent_operator", size_dofs, size_dofs);
    TangentOperator(delta_gen_coords, h, tangent_operator);

    Kokkos::deep_copy(iteration_matrix, 0.0);

    // Calculate the beam element static iteration matrix
    auto iteration_matrix_local = View2D("iteration_matrix_local", size_dofs, size_dofs);
    ElementalStaticStiffnessMatrix(
        position_vectors_, gen_coords, stiffness_matrix_, quadrature_, iteration_matrix_local
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
        View2D("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    auto temp = gen_alpha_solver::transpose_matrix(constraints_gradient_matrix);
    Kokkos::deep_copy(quadrant_2, temp);

    // quadrant_3 = B(q) * T(h dq)
    auto quadrant_3 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(size_dofs, size_iteration),
        Kokkos::make_pair(zero, size_dofs)
    );
    KokkosBlas::gemm("N", "N", 1.0, constraints_gradient_matrix, tangent_operator, 0.0, quadrant_3);
}

}  // namespace openturbine::gebt_poc
