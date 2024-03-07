#pragma once

#include <Kokkos_Core.hpp>

#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

/// Class to manage normalized location, mass matrix, and stiffness matrix of a beam section
class Section {
public:
    Section()
        : name_(""),
          location_(0.),
          mass_matrix_("mass matrix"),
          stiffness_matrix_("stiffness matrix") {}

    Section(std::string name, double location, View2D_6x6 mass_matrix, View2D_6x6 stiffness_matrix)
        : name_(name),
          location_(location),
          mass_matrix_(mass_matrix),
          stiffness_matrix_(stiffness_matrix) {
        if (location_ < 0. || location_ > 1.) {
            throw std::invalid_argument("Section location must be between 0 and 1");
        }
    }

    /// Returns the name of the section
    inline std::string GetName() const { return name_; }

    /// Returns the normalized location of the section
    inline double GetNormalizedLocation() const { return location_; }

    /// Returns the mass matrix of the section
    inline const View2D_6x6 GetMassMatrix() const { return mass_matrix_; }

    /// Returns the stiffness matrix of the section
    inline const View2D_6x6 GetStiffnessMatrix() const { return stiffness_matrix_; }

private:
    std::string name_;             //< Name of the section
    double location_;              //< Normalized location of the section (0 <= l <= 1)
    View2D_6x6 mass_matrix_;       //< Mass matrix of the section
    View2D_6x6 stiffness_matrix_;  //< Stiffness matrix of the section
};

}  // namespace openturbine::gebt_poc
