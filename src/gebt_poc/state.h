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
}  // namespace openturbine::gebt_poc
