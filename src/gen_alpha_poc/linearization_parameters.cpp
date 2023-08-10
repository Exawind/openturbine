#include "src/gen_alpha_poc/linearization_parameters.h"

namespace openturbine::gen_alpha_solver {

HostView1D UnityLinearizationParameters::ResidualVector(
    [[maybe_unused]] const HostView1D gen_coords, [[maybe_unused]] const HostView1D velocity,
    const HostView1D acceleration, const HostView1D lagrange_mults
) {
    auto size = acceleration.size() + lagrange_mults.size();
    return create_identity_vector(size);
}

HostView2D UnityLinearizationParameters::IterationMatrix(
    [[maybe_unused]] const double& h, [[maybe_unused]] const double& BETA_PRIME,
    [[maybe_unused]] const double& GAMMA_PRIME, [[maybe_unused]] const HostView1D gen_coords,
    [[maybe_unused]] const HostView1D delta_gen_coords, [[maybe_unused]] const HostView1D velocity,
    [[maybe_unused]] const HostView1D acceleration, const HostView1D lagrange_mults
) {
    auto size = velocity.size() + lagrange_mults.size();
    return create_identity_matrix(size);
}

}  // namespace openturbine::gen_alpha_solver
