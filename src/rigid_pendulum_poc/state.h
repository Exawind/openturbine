#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;
using HostView2D = Kokkos::View<double**, Kokkos::HostSpace>;

/// @brief A class to store and manage the states of a dynamic system
// TODO: Refactor this class to avoid/minimize expensive copies
class State {
public:
    State();
    State(HostView1D, HostView1D, HostView1D, HostView1D);

    /// Get the generalized coordinates
    inline HostView1D GetGeneralizedCoordinates() const { return generalized_coords_; }

    /// Get the first time derivative of the generalized coordinates
    inline HostView1D GetGeneralizedVelocity() const { return generalized_velocity_; }

    /// Get the second time derivative of the generalized coordinates
    inline HostView1D GetGeneralizedAcceleration() const { return generalized_acceleration_; }

    /// Get the algorithmic accelerations (different than the generalized accelerations)
    inline HostView1D GetAlgorithmicAcceleration() const { return algorithmic_acceleration_; }

private:
    HostView1D generalized_coords_;
    HostView1D generalized_velocity_;
    HostView1D generalized_acceleration_;
    HostView1D algorithmic_acceleration_;
};

}  // namespace openturbine::rigid_pendulum
