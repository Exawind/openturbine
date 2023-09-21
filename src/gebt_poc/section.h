#pragma once

#include "src/gen_alpha_poc/state.h"

namespace openturbine::gebt_poc {

/// Class to create and store a 6 x 6 stiffness matrix for a section
class StiffnessMatrix {
public:
    /// Default constructor that initializes the stiffness matrix to the identity matrix
    StiffnessMatrix();

    /// Constructor that initializes the stiffness matrix to the given matrix
    StiffnessMatrix(const Kokkos::View<double**>);

    /// Returns the stiffness matrix as read-only 2D Kokkos view
    inline Kokkos::View<const double**> GetStiffnessMatrix() const { return stiffness_matrix_; }

private:
    Kokkos::View<double**> stiffness_matrix_;  //< Stiffness matrix (6 x 6)
};

/// Class to manage normalized location, mass matrix, and stiffness matrix of a beam section
class Section {
public:
    Section() : name_(""), location_(0.), mass_matrix_(), stiffness_matrix_() {}

    Section(std::string name, double location, gen_alpha_solver::MassMatrix, StiffnessMatrix);

    /// Returns the name of the section
    inline std::string GetName() const { return name_; }

    /// Returns the normalized location of the section
    inline double GetNormalizedLocation() const { return location_; }

    /// Returns the mass matrix of the section
    inline const gen_alpha_solver::MassMatrix& GetMassMatrix() const { return mass_matrix_; }

    /// Returns the stiffness matrix of the section
    inline const StiffnessMatrix& GetStiffnessMatrix() const { return stiffness_matrix_; }

private:
    std::string name_;                          //< Name of the section
    double location_;                           //< Normalized location of the section (0 <= l <= 1)
    gen_alpha_solver::MassMatrix mass_matrix_;  //< Mass matrix of the section
    StiffnessMatrix stiffness_matrix_;          //< Stiffness matrix of the section
};

}  // namespace openturbine::gebt_poc
