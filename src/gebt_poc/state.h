#pragma once

#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gebt_poc {

/// @brief Class to store and manage the states of a dynamic system
class State {
public:
    /// Default constructor that initializes all states to zero (assuming a single node)
    State();

    /// Constructor that initializes the states to the given 2D Kokkos views
    State(
        LieGroupFieldView gen_coords, LieAlgebraFieldView velocity, LieAlgebraFieldView accln,
        LieAlgebraFieldView algo_accln
    );

    /// Returns the generalized coordinates vector
    auto GetGeneralizedCoordinates() const { return generalized_coords_; }

    /// Returns the velocity vector
    auto GetVelocity() const { return velocity_; }

    /// Returns the acceleration vector
    auto GetAcceleration() const { return acceleration_; }

    /// Returns the algorithmic acceleration vector
    auto GetAlgorithmicAcceleration() const { return algorithmic_acceleration_; }

    /// Returns the number of nodes in the system
    inline size_t GetNumberOfNodes() const { return generalized_coords_.extent(0); }

    // clang-format off
private :
    LieGroupFieldView generalized_coords_;
    LieAlgebraFieldView velocity_;
    LieAlgebraFieldView acceleration_;
    LieAlgebraFieldView algorithmic_acceleration_;
    // clang-format on
};

/// Class to create and store a 6 x 6 mass matrix of a rigid body
class MassMatrix {
public:
    /// Constructor that initializes the mass matrix with the given mass, moments of inertia,
    /// and center of mass
    MassMatrix(double mass, View1D_Vector center_of_mass, View2D_3x3 moment_of_inertia);

    /// Constructor that initializes the mass matrix with the given mass matrix
    MassMatrix(View2D_6x6 mass_matrix);

    /// Returns the mass matrix of the rigid body
    auto GetMassMatrix() const { return mass_matrix_; }

    /// Returns the mass of the rigid body
    inline double GetMass() const { return mass_; }

    /// Returns the center of mass of the rigid body
    auto GetCenterOfMass() const { return center_of_mass_; }

    /// Returns the moments of inertia of the rigid body
    auto GetMomentOfInertia() const { return moment_of_inertia_; }

private:
    double mass_;                   //< Mass of the rigid body
    View1D_Vector center_of_mass_;  //< Center of mass of the rigid body
    View2D_3x3 moment_of_inertia_;  //< Moments of inertia
    View2D_6x6 mass_matrix_;        //< Mass matrix of the rigid body
};

}  // namespace openturbine::gebt_poc
