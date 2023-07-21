#include "src/rigid_pendulum_poc/state.h"

namespace openturbine::rigid_pendulum {

State::State()
    : generalized_coords_("generalized_coordinates", 1),
      velocity_("velocities", 1),
      acceleration_("accelerations", 1),
      algorithmic_acceleration_("algorithmic_accelerations", 1) {
}

State::State(HostView1D q, HostView1D v, HostView1D v_dot, HostView1D a)
    : generalized_coords_("generalized_coordinates", q.size()),
      velocity_("velocities", v.size()),
      acceleration_("accelerations", v_dot.size()),
      algorithmic_acceleration_("algorithmic_accelerations", a.size()) {
    Kokkos::deep_copy(generalized_coords_, q);
    Kokkos::deep_copy(velocity_, v);
    Kokkos::deep_copy(acceleration_, v_dot);
    Kokkos::deep_copy(algorithmic_acceleration_, a);
}

}  // namespace openturbine::rigid_pendulum
