#pragma once

#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gen_alpha_solver {

// Used to store a 1D Kokkos View of doubles on the host
using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;

// Used to store a 2D Kokkos View of doubles on the host
using HostView2D = Kokkos::View<double**, Kokkos::HostSpace>;

/// @brief Class to store and manage the states of a dynamic system
class State {
public:
    /// Default constructor that initializes all states to zero with size one
    State();

    State(
        HostView1D generalized_coords_, HostView1D velocity_, HostView1D acceleration_,
        HostView1D algorithmic_acceleration_
    );

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

// TODO Move the following classes to their own source files

/// Class to create and store a 6 x 6 mass matrix of a rigid body
class MassMatrix {
public:
    /// Constructor that initializes the mass matrix with the given mass and moments of inertia
    MassMatrix(double mass, Vector principal_moment_of_inertia);

    /// Constructor that initializes the mass matrix with the same moment of inertia about all axes
    MassMatrix(double mass = 1., double moment_of_inertia = 1.);

    /// Constructor that initializes the mass matrix to the given matrix
    MassMatrix(HostView2D);

    /// Returns the mass of the rigid body
    inline double GetMass() const { return mass_; }

    /// Returns the principal moments of inertia of the rigid body
    inline Vector GetPrincipalMomentsOfInertia() const { return principal_moment_of_inertia_; }

    /// Returns the moment of inertia matrix as a 2D Kokkos view
    HostView2D GetMomentOfInertiaMatrix() const;

    /// Returns the mass matrix of the rigid body
    inline HostView2D GetMassMatrix() const { return mass_matrix_; }

private:
    double mass_;                         //< Mass of the rigid body
    Vector principal_moment_of_inertia_;  //< Moments of inertia about the principal axes

    HostView2D mass_matrix_;  //< Mass matrix of the rigid body
};

/// Class for managing the generalized forces applied on a dynamic system
class GeneralizedForces {
public:
    /// Default constructor that initializes all generalized forces to zero
    GeneralizedForces(const Vector& forces = Vector(), const Vector& moments = Vector());

    /// Constructor that initializes the generalized forces to the given vectors
    GeneralizedForces(HostView1D);

    /// Returns the force vector
    inline Vector GetForces() const { return forces_; }

    /// Returns the moment vector
    inline Vector GetMoments() const { return moments_; }

    /// Returns the generalized forces vector
    inline HostView1D GetGeneralizedForces() const { return generalized_forces_; }

private:
    Vector forces_;   //< force vector
    Vector moments_;  //< moment vector

    HostView1D generalized_forces_;  //< Generalized forces (combined forces and moments vector)
};

}  // namespace openturbine::gen_alpha_solver
