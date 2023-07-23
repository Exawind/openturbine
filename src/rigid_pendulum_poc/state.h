#pragma once

#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

/// @brief Class to store and manage the states of a dynamic system
class State {
public:
    /// Default constructor that initializes all states to zero with size one
    State();

    State(HostView1D, HostView1D, HostView1D, HostView1D);

    /// Returns the generalized coordinates vector
    inline HostView1D GetGeneralizedCoordinates() const { return generalized_coords_; }

    /// Returns the velocity vector
    inline HostView1D GetVelocity() const { return velocity_; }

    /// Returns the acceleration vector
    inline HostView1D GetAcceleration() const { return acceleration_; }

    /// Returns the algorithmic accelerations vector
    inline HostView1D GetAlgorithmicAcceleration() const { return algorithmic_acceleration_; }

private:
    HostView1D generalized_coords_;        //< Generalized coordinates
    HostView1D velocity_;                  //< Velocity vector
    HostView1D acceleration_;              //< First time derivative of the velocity vector
    HostView1D algorithmic_acceleration_;  //< Algorithmic accelerations
};

}  // namespace openturbine::rigid_pendulum
