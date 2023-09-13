#include "src/gebt_poc/section.h"

namespace openturbine::gebt_poc {

StiffnessMatrix::StiffnessMatrix(const Kokkos::View<double**> stiffness)
    : stiffness_matrix_("stiffness_matrix", stiffness.extent(0), stiffness.extent(1)) {
    if (stiffness_matrix_.extent(0) != 6 || stiffness_matrix_.extent(1) != 6) {
        throw std::invalid_argument("Stiffness matrix must be 6 x 6");
    }

    Kokkos::deep_copy(stiffness_matrix_, stiffness);
}

}  // namespace openturbine::gebt_poc
