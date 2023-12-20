#include "src/gebt_poc/dynamic_beam_element.h"

#include <iostream>

#include <KokkosBlas.hpp>

#include "src/gebt_poc/static_beam_element.h"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

DynamicBeamLinearizationParameters::DynamicBeamLinearizationParameters(
    Kokkos::View<double*> position_vectors, StiffnessMatrix stiffness_matrix,
    UserDefinedQuadrature quadrature
)
    : position_vectors_(position_vectors),
      stiffness_matrix_(stiffness_matrix),
      quadrature_(quadrature) {
}

void DynamicBeamLinearizationParameters::ResidualVector(
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double*> residual
) {
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
