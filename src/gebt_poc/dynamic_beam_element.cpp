#include "src/gebt_poc/dynamic_beam_element.h"

#include <iostream>

#include <KokkosBlas.hpp>

#include "src/gebt_poc/static_beam_element.h"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

DynamicBeamLinearizationParameters::DynamicBeamLinearizationParameters(
    Kokkos::View<double*> position_vectors, StiffnessMatrix stiffness_matrix, MassMatrix mass_matrix,
    UserDefinedQuadrature quadrature, std::vector<GeneralizedForces> external_forces
)
    : position_vectors_(position_vectors),
      stiffness_matrix_(stiffness_matrix),
      mass_matrix_(mass_matrix),
      quadrature_(quadrature),
      external_forces_(std::move(external_forces)) {
}

void DynamicBeamLinearizationParameters::ApplyExternalForces(
    const std::vector<GeneralizedForces>& generalized_forces, Kokkos::View<double*> external_forces
) {
    Kokkos::deep_copy(external_forces, 0.0);
    for (const auto& force : generalized_forces) {
        auto gen_forces = force.GetGeneralizedForces();
        auto node = force.GetNode();
        auto external_forces_node = Kokkos::subview(
            external_forces,
            Kokkos::make_pair(
                (node - 1) * kNumberOfLieAlgebraComponents, node * kNumberOfLieAlgebraComponents
            )
        );
        Kokkos::parallel_for(
            kNumberOfLieAlgebraComponents,
            KOKKOS_LAMBDA(size_t i) { external_forces_node(i) = gen_forces(i); }
        );
    }
}

void DynamicBeamLinearizationParameters::ResidualVector(
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double*> residual,
    const gen_alpha_solver::TimeStepper& time_stepper
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

    auto gen_coords_1D =
        Kokkos::View<double*>("gen_coords_1D", gen_coords.extent(0) * gen_coords.extent(1));
    Convert2DViewTo1DView(gen_coords, gen_coords_1D);
    auto velocity_1D = Kokkos::View<double*>("velocity_1D", velocity.extent(0) * velocity.extent(1));
    Convert2DViewTo1DView(velocity, velocity_1D);
    auto acceleration_1D =
        Kokkos::View<double*>("acceleration_1D", acceleration.extent(0) * acceleration.extent(1));
    Convert2DViewTo1DView(acceleration, acceleration_1D);

    // Assemble the top partition of the residual vector consisting of 4 parts
    // Part 1: elastic/static forces residual
    auto residual_elastic = Kokkos::subview(residual, Kokkos::make_pair(zero, size_dofs));
    ElementalStaticForcesResidual(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, residual_elastic
    );
    // Part 2: dynamic/inertial forces residual
    auto residual_inertial = Kokkos::View<double*>("residual_inertial", residual_elastic.extent(0));
    ElementalInertialForcesResidual(
        position_vectors_, gen_coords_1D, velocity_1D, acceleration_1D, mass_matrix_, quadrature_,
        residual_inertial
    );
    KokkosBlas::axpy(1., residual_inertial, residual_elastic);
    // Part 3: external forces
    auto external_forces = Kokkos::View<double*>("external_forces", residual_elastic.extent(0));
    ApplyExternalForces(this->external_forces_, external_forces);
    KokkosBlas::axpy(-1., external_forces, residual_elastic);
    // Part 4: Calculate the contribution for the constraints
    auto constraints_gradient_matrix =
        Kokkos::View<double**>("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    auto constraints_part2 = Kokkos::View<double*>("constraints_part2", size_dofs);
    KokkosBlas::gemv(
        "T", 1., constraints_gradient_matrix, lagrange_multipliers, 0., constraints_part2
    );
    KokkosBlas::axpy(1., constraints_part2, residual_elastic);

    // Assemble the bottom partition of the residual vector i.e. the constraints residual
    auto residual_constraints =
        Kokkos::subview(residual, Kokkos::make_pair(size_dofs, size_residual));
    ElementalConstraintForcesResidual(gen_coords_1D, residual_constraints);
}

void DynamicBeamLinearizationParameters::IterationMatrix(
    const double& h, const double& beta_prime, const double& gamma_prime,
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double**> iteration_matrix
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
    const auto n_nodes = velocity.extent(0);

    auto gen_coords_1D =
        Kokkos::View<double*>("gen_coords_1D", gen_coords.extent(0) * gen_coords.extent(1));
    Convert2DViewTo1DView(gen_coords, gen_coords_1D);
    auto velocity_1D = Kokkos::View<double*>("velocity_1D", velocity.extent(0) * velocity.extent(1));
    Convert2DViewTo1DView(velocity, velocity_1D);
    auto accelaration_1D =
        Kokkos::View<double*>("accelaration_1D", acceleration.extent(0) * acceleration.extent(1));
    Convert2DViewTo1DView(acceleration, accelaration_1D);

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

    // Assemble the top left partition of the iteration matrix consisting of 3 parts
    auto mass_matrix = Kokkos::View<double**>("mass_matrix", size_dofs, size_dofs);
    auto gyroscopic_matrix = Kokkos::View<double**>("gyroscopic_matrix", size_dofs, size_dofs);
    auto dynamic_stiffness_matrix =
        Kokkos::View<double**>("dynamic_stiffness_matrix", size_dofs, size_dofs);
    ElementalInertialMatrices(
        position_vectors_, gen_coords_1D, velocity_1D, accelaration_1D, mass_matrix_, quadrature_,
        mass_matrix, gyroscopic_matrix, dynamic_stiffness_matrix
    );
    // Part 1: [M] * beta'
    auto quadrant_1 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs), Kokkos::make_pair(zero, size_dofs)
    );
    KokkosBlas::axpy(beta_prime, mass_matrix, quadrant_1);
    // Part 2: [G] * gamma'
    auto G_gamma = Kokkos::View<double**>("G_gamma", size_dofs, size_dofs);
    KokkosBlas::axpy(gamma_prime, gyroscopic_matrix, G_gamma);
    KokkosBlas::axpy(1., G_gamma, quadrant_1);
    // Part 3: [K_t(q,v,v',Lambda,t)] * [T(h dq)]
    // K_t = K_t_elastic + K_t_inertial
    auto stiffness_matrix = Kokkos::View<double**>("stiffness_matrix", size_dofs, size_dofs);
    ElementalStaticStiffnessMatrix(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, stiffness_matrix
    );
    KokkosBlas::axpy(1., dynamic_stiffness_matrix, stiffness_matrix);
    auto K_t_T = Kokkos::View<double**>("K_t_T", size_dofs, size_dofs);
    KokkosBlas::gemm("N", "N", 1.0, stiffness_matrix, tangent_operator, 0.0, K_t_T);
    KokkosBlas::axpy(1., K_t_T, quadrant_1);

    // Top right partition of the iteration matrix = Transpose([B(q)])
    auto quadrant_2 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs),
        Kokkos::make_pair(size_dofs, size_iteration)
    );
    auto constraints_gradient_matrix =
        Kokkos::View<double**>("constraints_gradient_matrix", size_constraints, size_dofs);
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
