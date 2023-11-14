#include "src/gebt_poc/section.h"

namespace openturbine::gebt_poc {

void init(Kokkos::View<double[6][6]> stiffness_matrix) {
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(std::size_t) {
      gen_alpha_solver::create_identity_matrix<Kokkos::DefaultExecutionSpace>(stiffness_matrix);
    });
}

StiffnessMatrix::StiffnessMatrix() : stiffness_matrix_("stiffness_matrix", 6, 6) {
    init(stiffness_matrix_);
}

StiffnessMatrix::StiffnessMatrix(const Kokkos::View<double**> stiffness)
    : stiffness_matrix_("stiffness_matrix", stiffness.extent(0), stiffness.extent(1)) {
    if (stiffness_matrix_.extent(0) != 6 || stiffness_matrix_.extent(1) != 6) {
        throw std::invalid_argument("Stiffness matrix must be 6 x 6");
    }

    Kokkos::deep_copy(stiffness_matrix_, stiffness);
}

Section::Section(
    std::string name, double location, gen_alpha_solver::MassMatrix mass_matrix,
    StiffnessMatrix stiffness_matrix
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
