#include "src/gebt_poc/dynamic_beam_element.h"

#include <iostream>

#include <KokkosBlas.hpp>

#include "src/gebt_poc/static_beam_element.h"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

DynamicBeamLinearizationParameters::DynamicBeamLinearizationParameters(
    Kokkos::View<double*> position_vectors, StiffnessMatrix stiffness_matrix, MassMatrix mass_matrix,
    UserDefinedQuadrature quadrature
)
    : position_vectors_(position_vectors),
      stiffness_matrix_(stiffness_matrix),
      mass_matrix_(mass_matrix),
      quadrature_(quadrature) {
}

void DynamicBeamLinearizationParameters::ResidualVector(
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double*> residual
) {
    // Residual vector for a dynamic beam element is assembled as follows
    // {residual} = {
    //     {residual_elastic} + {residual_inertial} + {constraints}
    //     {residual_constraints}
    // }
    // where,
    // {residual_elastic} = elastic/static forces residual vector
    // {residual_inertial} = dynamic/inertial forces residual vector
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

    // Assemble the top partition of the residual vector consisting of 3 parts
    // Part 1: elastic/static forces residual
    auto residual_elastic = Kokkos::subview(residual, Kokkos::make_pair(zero, size_dofs));
    ElementalStaticForcesResidual(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, residual_elastic
    );
    // Part 2: dynamic/inertial forces residual
    // void ElementalInertialForcesResidual(
    //     const Kokkos::View<double*> position_vectors, const Kokkos::View<double*> gen_coords,
    //     const Kokkos::View<double*> velocity, const Kokkos::View<double*> acceleration,
    //     const MassMatrix& mass_matrix, const Quadrature& quadrature, Kokkos::View<double*>
    //     residual
    // );
    auto residual_inertial = Kokkos::View<double*>("residual_inertial", residual_elastic.extent(0));
    ElementalInertialForcesResidual(
        position_vectors_, gen_coords_1D, velocity_1D, acceleration_1D, mass_matrix_, quadrature_,
        residual_inertial
    );
    KokkosBlas::axpy(1., residual_inertial, residual_elastic);
    // Part 3: Calculate the contribution for the constraints
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
    const double& h, [[maybe_unused]] const double& beta_prime,
    [[maybe_unused]] const double& gamma_prime,
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double**> iteration_matrix
) {
}

}  // namespace openturbine::gebt_poc
