#pragma once

#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gebt_poc {

/// @brief Class to store and manage the states of a dynamic system
class State {
public:
    static constexpr size_t kNumberOfLieGroupComponents = 7;
    static constexpr size_t kNumberOfLieAlgebraComponents = 6;
    static constexpr size_t kNumberOfVectorComponents = 3;

    /// Default constructor that initializes all states to zero (assuming a single node)
    State();

    /// Constructor that initializes the states to the given 2D Kokkos views
    State(
        Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> accln,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> algo_accln
    );

    /// Returns the generalized coordinates vector
    inline Kokkos::View<double* [kNumberOfLieGroupComponents]> GetGeneralizedCoordinates() const {
        return generalized_coords_;
    }

    /// Returns the velocity vector
    inline Kokkos::View<double* [kNumberOfLieAlgebraComponents]> GetVelocity() const {
        return velocity_;
    }

    /// Returns the acceleration vector
    inline Kokkos::View<double* [kNumberOfLieAlgebraComponents]> GetAcceleration() const {
        return acceleration_;
    }

    /// Returns the algorithmic acceleration vector
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
    /// Constructor that initializes the mass matrix with the given mass, moments of inertia,
    /// and center of mass
    MassMatrix(
        double mass, Kokkos::View<double[3]> center_of_mass,
        Kokkos::View<double[3][3]> moment_of_inertia
    );

    /// Constructor that initializes the mass matrix with the given mass matrix
    MassMatrix(Kokkos::View<double[6][6]> mass_matrix);

    /// Returns the mass matrix of the rigid body
    inline Kokkos::View<double**> GetMassMatrix() const { return mass_matrix_; }

    /// Returns the mass of the rigid body
    inline double GetMass() const { return mass_; }

    /// Returns the center of mass of the rigid body
    inline Kokkos::View<double[3]> GetCenterOfMass() const { return center_of_mass_; }

    /// Returns the moments of inertia of the rigid body
    inline Kokkos::View<double[3][3]> GetMomentOfInertia() const { return moment_of_inertia_; }

private:
    double mass_;                                   //< Mass of the rigid body
    Kokkos::View<double[3]> center_of_mass_;        //< Center of mass of the rigid body
    Kokkos::View<double[3][3]> moment_of_inertia_;  //< Moments of inertia
    Kokkos::View<double[6][6]> mass_matrix_;        //< Mass matrix of the rigid body
};

/// Class for managing the generalized forces applied on a dynamic system
class GeneralizedForces {
public:
    /// Default constructor that initializes all generalized forces to zero
    GeneralizedForces(
        const gen_alpha_solver::Vector& forces = gen_alpha_solver::Vector(),
        const gen_alpha_solver::Vector& moments = gen_alpha_solver::Vector()
    );

    /// Constructor that initializes the generalized forces to the given vectors
    GeneralizedForces(Kokkos::View<double[6]>);

    /// Returns the 3 x 1 force vector
    inline gen_alpha_solver::Vector GetForces() const { return forces_; }

    /// Returns the 3 x 1 moment vector
    inline gen_alpha_solver::Vector GetMoments() const { return moments_; }

    /// Returns the 6 x 1 generalized forces vector
    inline Kokkos::View<double*> GetGeneralizedForces() const { return generalized_forces_; }

private:
    gen_alpha_solver::Vector forces_;   //< force vector
    gen_alpha_solver::Vector moments_;  //< moment vector
    Kokkos::View<double[6]>
        generalized_forces_;  //< Generalized forces (combined forces and moments vector)
};

}  // namespace openturbine::gebt_poc
