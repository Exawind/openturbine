#include "src/rigid_pendulum_poc/state.h"

namespace openturbine::rigid_pendulum {

State::State()
    : generalized_coords_("generalized_coordinates", 1),
      generalized_velocity_("generalized_velocity", 1),
      generalized_acceleration_("generalized_accelerations", 1),
      algorithmic_acceleration_("algorithmic_accelerations", 1) {
}

State::State(
    HostView1D gen_coords, HostView1D gen_velocity, HostView1D gen_accln, HostView1D algo_accln
)
    : generalized_coords_("generalized_coordinates", gen_coords.size()),
      generalized_velocity_("generalized_velocity", gen_velocity.size()),
      generalized_acceleration_("generalized_accelerations", gen_accln.size()),
      algorithmic_acceleration_("algorithmic_accelerations", algo_accln.size()) {
    Kokkos::deep_copy(generalized_coords_, gen_coords);
    Kokkos::deep_copy(generalized_velocity_, gen_velocity);
    Kokkos::deep_copy(generalized_acceleration_, gen_accln);
    Kokkos::deep_copy(algorithmic_acceleration_, algo_accln);
}

}  // namespace openturbine::rigid_pendulum
