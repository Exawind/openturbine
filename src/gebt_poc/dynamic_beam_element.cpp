#include "src/gebt_poc/dynamic_beam_element.h"

#include <iostream>

#include <KokkosBlas.hpp>

#include "src/gebt_poc/static_beam_element.h"
#include "src/gebt_poc/types.hpp"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

DynamicBeamLinearizationParameters::DynamicBeamLinearizationParameters(
    LieGroupFieldView position_vectors, StiffnessMatrix stiffness_matrix, MassMatrix mass_matrix,
    UserDefinedQuadrature quadrature, std::vector<Forces*> external_forces
)
    : position_vectors_(std::move(position_vectors)),
      stiffness_matrix_(std::move(stiffness_matrix)),
      mass_matrix_(std::move(mass_matrix)),
      quadrature_(std::move(quadrature)),
      external_forces_(std::move(external_forces)) {
}

void DynamicBeamLinearizationParameters::ApplyExternalForces(
    double time, const Kokkos::View<double*>& external_forces
) {
    Kokkos::deep_copy(external_forces, 0.0);
    for (const auto* force : this->external_forces_) {
        auto gen_forces = force->GetGeneralizedForces(time);
        auto node = force->GetNode();
        auto external_forces_node = Kokkos::subview(
            external_forces,
            Kokkos::make_pair((node - 1) * LieAlgebraComponents, node * LieAlgebraComponents)
        );
        Kokkos::parallel_for(
            LieAlgebraComponents, KOKKOS_LAMBDA(size_t component
                                  ) { external_forces_node(component) = gen_forces(component); }
        );
    }
}

void DynamicBeamLinearizationParameters::ResidualVector(
    LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type velocity,
    LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_multipliers,
    const gen_alpha_solver::TimeStepper& time_stepper, View1D residual
) {
    // Residual vector for a dynamic beam element is assembled as follows
    // {residual} = {
    //     {residual_elastic} + {residual_inertial} - {external_forces} + {constraints}
    //     {residual_constraints}
    // }
    // where,
    // {residual_elastic} = elastic/static forces residual vector
    // {residual_inertial} = dynamic/inertial forces residual vector
    // {external_forces} = external forces applied on the beam element
    // {constraints} = [constraints_gradient_matrix] * {lagrange_multipliers}
    // {residual_constraints} = constraint forces residual vector

    Kokkos::deep_copy(residual, 0.0);

    const size_t zero{0};
    const auto size_dofs = velocity.extent(0) * velocity.extent(1);
    const auto size_constraints = lagrange_multipliers.extent(0);
    const auto size_residual = size_dofs + size_constraints;

    // Assemble the top partition of the residual vector consisting of 4 parts
    // Part 1: elastic/static forces residual
    auto residual_elastic = Kokkos::subview(residual, Kokkos::make_pair(zero, size_dofs));
    ElementalStaticForcesResidual(
        position_vectors_, gen_coords, stiffness_matrix_, quadrature_, residual_elastic
    );
    // Part 2: dynamic/inertial forces residual
    auto residual_inertial = View1D("residual_inertial", residual_elastic.extent(0));
    ElementalInertialForcesResidual(
        position_vectors_, gen_coords, velocity, acceleration, mass_matrix_, quadrature_,
        residual_inertial
    );
    KokkosBlas::axpy(1., residual_inertial, residual_elastic);
    // Part 3: external forces
    auto external_forces = View1D("external_forces", residual_elastic.extent(0));
    ApplyExternalForces(time_stepper.GetCurrentTime(), external_forces);
    KokkosBlas::axpy(-1., external_forces, residual_elastic);
    // Part 4: Calculate the contribution for the constraints
    auto constraints_gradient_matrix =
        View2D("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    auto constraints_part2 = View1D("constraints_part2", size_dofs);
    KokkosBlas::gemv(
        "T", 1., constraints_gradient_matrix, lagrange_multipliers, 0., constraints_part2
    );
    KokkosBlas::axpy(1., constraints_part2, residual_elastic);

    // Assemble the bottom partition of the residual vector i.e. the constraints residual
    auto residual_constraints =
        Kokkos::subview(residual, Kokkos::make_pair(size_dofs, size_residual));
    ElementalConstraintForcesResidual(gen_coords, residual_constraints);
}

void DynamicBeamLinearizationParameters::IterationMatrix(
    double h, double beta_prime, double gamma_prime, LieGroupFieldView::const_type gen_coords,
    LieAlgebraFieldView::const_type delta_gen_coords, LieAlgebraFieldView::const_type velocity,
    LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_multipliers,
    View2D iteration_matrix
) {
    // Iteration matrix for the static beam element is given by
    // [iteration matrix] = [
    //     [M] * beta' + G * gamma' + [K_t(q,v,v',Lambda,t)] * [T(h dq)]       [B(q)^T]]
    //                   [B(q)] * [T(h dq)]                                       [0]
    // ]
    // where,
    // [M] = Mass matrix of the beam element
    // beta' = beta prime (scalar)
    // [G] = Gyroscopic matrix of the beam element
    // gamma' = gamma prime (scalar)
    // [K_t(q,v,v',Lambda,t)] = Tangent stiffness matrix of the beam element
    // [T(h dq)] = Tangent operator
    // [B(q)] = Constraint gradient matrix

    Kokkos::deep_copy(iteration_matrix, 0.0);

    const size_t zero{0};
    const auto size_dofs = velocity.extent(0) * velocity.extent(1);
    const auto size_constraints = lagrange_multipliers.extent(0);
    const auto size_iteration = size_dofs + size_constraints;
    // const auto n_nodes = velocity.extent(0);

    // Assemble the tangent operator (same size as the stiffness matrix)
    auto delta_gen_coords_node = View1D_Vector("delta_gen_coords_node");
    auto tangent_operator = View2D("tangent_operator", size_dofs, size_dofs);
    Kokkos::deep_copy(tangent_operator, 0.0);
    TangentOperator(delta_gen_coords, h, tangent_operator);

    // Assemble the top left partition of the iteration matrix consisting of 3 parts
    auto mass_matrix = View2D("mass_matrix", size_dofs, size_dofs);
    auto gyroscopic_matrix = View2D("gyroscopic_matrix", size_dofs, size_dofs);
    auto dynamic_stiffness_matrix = View2D("dynamic_stiffness_matrix", size_dofs, size_dofs);
    ElementalInertialMatrices(
        position_vectors_, gen_coords, velocity, acceleration, mass_matrix_, quadrature_,
        mass_matrix, gyroscopic_matrix, dynamic_stiffness_matrix
    );
    // Part 1: [M] * beta'
    auto quadrant_1 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs), Kokkos::make_pair(zero, size_dofs)
    );
    KokkosBlas::axpy(beta_prime, mass_matrix, quadrant_1);
    // Part 2: [G] * gamma'
    auto G_gamma = View2D("G_gamma", size_dofs, size_dofs);
    KokkosBlas::axpy(gamma_prime, gyroscopic_matrix, G_gamma);
    KokkosBlas::axpy(1., G_gamma, quadrant_1);
    // Part 3: [K_t(q,v,v',Lambda,t)] * [T(h dq)]
    // K_t = K_t_elastic + K_t_inertial
    auto stiffness_matrix = View2D("stiffness_matrix", size_dofs, size_dofs);
    ElementalStaticStiffnessMatrix(
        position_vectors_, gen_coords, stiffness_matrix_, quadrature_, stiffness_matrix
    );
    KokkosBlas::axpy(1., dynamic_stiffness_matrix, stiffness_matrix);
    auto K_t_T = View2D("K_t_T", size_dofs, size_dofs);
    KokkosBlas::gemm("N", "N", 1.0, stiffness_matrix, tangent_operator, 0.0, K_t_T);
    KokkosBlas::axpy(1., K_t_T, quadrant_1);

    // Top right partition of the iteration matrix = Transpose([B(q)])
    auto quadrant_2 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs),
        Kokkos::make_pair(size_dofs, size_iteration)
    );
    auto constraints_gradient_matrix =
        View2D("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    auto temp = gen_alpha_solver::transpose_matrix(constraints_gradient_matrix);
    Kokkos::deep_copy(quadrant_2, temp);

    // Bottom left partition of the iteration matrix = [B(q)] * [T(h dq)]
    auto quadrant_3 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(size_dofs, size_iteration),
        Kokkos::make_pair(zero, size_dofs)
    );
    KokkosBlas::gemm("N", "N", 1.0, constraints_gradient_matrix, tangent_operator, 0.0, quadrant_3);
}

}  // namespace openturbine::gebt_poc
