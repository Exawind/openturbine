#pragma once

#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gebt_poc {

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

}  // namespace openturbine::gebt_poc
