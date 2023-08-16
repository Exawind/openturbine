#include "src/rigid_pendulum_poc/linearization_parameters.h"

namespace openturbine::rigid_pendulum {

Kokkos::View<double*> UnityLinearizationParameters::ResidualVector(
    [[maybe_unused]] const Kokkos::View<double*> gen_coords, [[maybe_unused]] const Kokkos::View<double*> velocity,
    const Kokkos::View<double*> acceleration, const Kokkos::View<double*> lagrange_mults
) {
    auto size = acceleration.size() + lagrange_mults.size();
    return create_identity_vector(size);
}

Kokkos::View<double**> UnityLinearizationParameters::IterationMatrix(
    [[maybe_unused]] const double& h, [[maybe_unused]] const double& BETA_PRIME,
    [[maybe_unused]] const double& GAMMA_PRIME, [[maybe_unused]] const Kokkos::View<double*> gen_coords,
    [[maybe_unused]] const Kokkos::View<double*> delta_gen_coords, [[maybe_unused]] const Kokkos::View<double*> velocity,
    [[maybe_unused]] const Kokkos::View<double*> acceleration, const Kokkos::View<double*> lagrange_mults
) {
    auto size = velocity.size() + lagrange_mults.size();
    return create_identity_matrix(size);
}

}  // namespace openturbine::rigid_pendulum
