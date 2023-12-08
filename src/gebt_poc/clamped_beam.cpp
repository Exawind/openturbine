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
}

void ClampedBeamLinearizationParameters::TangentOperator(
    Kokkos::View<double[kNumberOfVectorComponents]> psi, Kokkos::View<double**> tangent_operator
) {
    auto populate_matrix = KOKKOS_LAMBDA(size_t) {
        tangent_operator(0, 0) = 1.;
        tangent_operator(1, 1) = 1.;
        tangent_operator(2, 2) = 1.;
        tangent_operator(3, 3) = 1.;
        tangent_operator(4, 4) = 1.;
        tangent_operator(5, 5) = 1.;
    };
    Kokkos::parallel_for(1, populate_matrix);
}

}  // namespace openturbine::gebt_poc
