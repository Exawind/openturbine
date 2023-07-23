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

/// Class to store and manage a 6 x 6 mass matrix of a rigid body
class MassMatrix {
public:
    /// Default constructor that initializes the mass matrix to the identity matrix
    MassMatrix(double mass = 1., double moment_of_inertia = 1.);

    /// Constructor that initializes the mass matrix to the given matrix
    MassMatrix(HostView2D);

    /// Returns the mass of the rigid body
    inline double GetMass() const { return mass_; }

    /// Returns the moment of inertia of the rigid body
    inline double GetMomentOfInertia() const { return moment_of_inertia_; }

    /// Returns the mass matrix of the rigid body
    inline HostView2D GetMassMatrix() const { return mass_matrix_; }

private:
    double mass_;               //< Mass of the rigid body
    double moment_of_inertia_;  //< Moment of inertia of the rigid body

    HostView2D mass_matrix_;  //< Mass matrix of the rigid body
};

}  // namespace openturbine::rigid_pendulum
