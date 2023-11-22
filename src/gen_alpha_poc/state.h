#pragma once

#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gen_alpha_solver {

/// @brief Class to store and manage the states of a dynamic system
class State {
public:
    static constexpr size_t kNumberOfLieGroupComponents = 7;
    static constexpr size_t kNumberOfLieAlgebraComponents = 6;

    /// Default constructor that initializes all states to zero (assuming a single node)
    State();

    /// Constructor that initializes the states to the given 2D Kokkos views
    State(
        Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> accln,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> algo_accln
    );

    /// Returns the generalized coordinates vector (read only)
    inline Kokkos::View<double* [kNumberOfLieGroupComponents]> GetGeneralizedCoordinates() const {
        return generalized_coords_;
    }

    /// Returns the velocity vector (read only)
    inline Kokkos::View<double* [kNumberOfLieAlgebraComponents]> GetVelocity() const {
        return velocity_;
    }

    /// Returns the acceleration vector (read only)
    inline Kokkos::View<double* [kNumberOfLieAlgebraComponents]> GetAcceleration() const {
        return acceleration_;
    }

    /// Returns the algorithmic acceleration vector (read only)
    inline Kokkos::View<double* [kNumberOfLieAlgebraComponents]> GetAlgorithmicAcceleration() const {
        return algorithmic_acceleration_;
    }

    /// Returns the number of nodes in the system
    inline size_t GetNumberOfNodes() const {
        return generalized_coords_.extent(0);
    }

    // clang-format off
private :
    Kokkos::View<double*[kNumberOfLieGroupComponents]> generalized_coords_;
    Kokkos::View<double*[kNumberOfLieAlgebraComponents]> velocity_;
    Kokkos::View<double*[kNumberOfLieAlgebraComponents]> acceleration_;
    Kokkos::View<double*[kNumberOfLieAlgebraComponents]> algorithmic_acceleration_;
    // clang-format on
};

/// Class to create and store a 6 x 6 mass matrix of a rigid body
class MassMatrix {
public:
    /// Constructor that initializes the mass matrix with the given mass and moments of inertia
    MassMatrix(double mass, Vector principal_moment_of_inertia);

    /// Constructor that initializes the mass matrix with the same moment of inertia about all axes
    MassMatrix(double mass = 1., double moment_of_inertia = 1.);

    /// Constructor that initializes the mass matrix to the given matrix
    MassMatrix(Kokkos::View<double**>);

    /// Returns the mass of the rigid body
    inline double GetMass() const { return mass_; }

    /// Returns the principal moments of inertia of the rigid body
    inline Vector GetPrincipalMomentsOfInertia() const { return principal_moment_of_inertia_; }

    /// Returns the moment of inertia matrix as a 2D Kokkos view
    Kokkos::View<double**> GetMomentOfInertiaMatrix() const;

    /// Returns the mass matrix of the rigid body
    inline Kokkos::View<double**> GetMassMatrix() const { return mass_matrix_; }

private:
    double mass_;                         //< Mass of the rigid body
    Vector principal_moment_of_inertia_;  //< Moments of inertia about the principal axes

    Kokkos::View<double**> mass_matrix_;  //< Mass matrix of the rigid body
};

/// Class for managing the generalized forces applied on a dynamic system
class GeneralizedForces {
public:
    /// Default constructor that initializes all generalized forces to zero
    GeneralizedForces(const Vector& forces = Vector(), const Vector& moments = Vector());

    /// Constructor that initializes the generalized forces to the given vectors
    GeneralizedForces(Kokkos::View<double*>);

    /// Returns the force vector
    inline Vector GetForces() const { return forces_; }

    /// Returns the moment vector
    inline Vector GetMoments() const { return moments_; }

    /// Returns the generalized forces vector
    inline Kokkos::View<double*> GetGeneralizedForces() const { return generalized_forces_; }

private:
    Vector forces_;   //< force vector
    Vector moments_;  //< moment vector

    Kokkos::View<double*>
        generalized_forces_;  //< Generalized forces (combined forces and moments vector)
};

}  // namespace openturbine::gen_alpha_solver
