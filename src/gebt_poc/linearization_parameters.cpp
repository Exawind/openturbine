#include "src/gebt_poc/linearization_parameters.h"

namespace openturbine::gebt_poc {

Kokkos::View<double*> UnityLinearizationParameters::ResidualVector(
    [[maybe_unused]] const Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    [[maybe_unused]] const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    const Kokkos::View<double*> lagrange_mults
) {
    auto size = acceleration.extent(0) * acceleration.extent(1) + lagrange_mults.extent(0);
    return gen_alpha_solver::create_identity_vector(size);
}

Kokkos::View<double**> UnityLinearizationParameters::IterationMatrix(
    [[maybe_unused]] const double& h, [[maybe_unused]] const double& BETA_PRIME,
    [[maybe_unused]] const double& GAMMA_PRIME,
    [[maybe_unused]] const Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    [[maybe_unused]] const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
    [[maybe_unused]] const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    const Kokkos::View<double*> lagrange_mults
) {
    auto size = velocity.extent(0) * velocity.extent(1) + lagrange_mults.extent(0);
    return gen_alpha_solver::create_identity_matrix(size);
}

}  // namespace openturbine::gebt_poc
