#pragma once

#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gebt_poc {

/// Class to create and store a 6 x 6 stiffness matrix for a section
class StiffnessMatrix {
public:
    /// Constructor that initializes the stiffness matrix to the given matrix
    StiffnessMatrix(const Kokkos::View<double**>);

    /// Returns the stiffness matrix as read-only 2D Kokkos view
    inline Kokkos::View<const double**> GetStiffnessMatrix() const { return stiffness_matrix_; }

private:
    Kokkos::View<double**> stiffness_matrix_;  //< Stiffness matrix
};

}  // namespace openturbine::gebt_poc
