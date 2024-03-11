#include "src/gebt_poc/section.h"

namespace openturbine::gebt_poc {

StiffnessMatrix::StiffnessMatrix() : stiffness_matrix_("stiffness_matrix", 6, 6) {
    stiffness_matrix_ = gen_alpha_solver::create_identity_matrix(6);
}

StiffnessMatrix::StiffnessMatrix(View2D::const_type stiffness)
    : stiffness_matrix_("stiffness_matrix", stiffness.extent(0), stiffness.extent(1)) {
    if (stiffness_matrix_.extent(0) != 6 || stiffness_matrix_.extent(1) != 6) {
        throw std::invalid_argument("Stiffness matrix must be 6 x 6");
    }

    Kokkos::deep_copy(stiffness_matrix_, stiffness);
}

Section::Section(
    std::string name, double location, MassMatrix mass_matrix, StiffnessMatrix stiffness_matrix
)
    : name_(name),
      location_(location),
      mass_matrix_(mass_matrix),
      stiffness_matrix_(stiffness_matrix) {
    if (location_ < 0. || location_ > 1.) {
        throw std::invalid_argument("Section location must be between 0 and 1");
    }
}

}  // namespace openturbine::gebt_poc
