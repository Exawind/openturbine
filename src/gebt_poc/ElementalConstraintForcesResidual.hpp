#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

inline void ElementalConstraintForcesResidual(
    LieGroupFieldView::const_type gen_coords, View1D constraints_residual
) {
    Kokkos::deep_copy(constraints_residual, 0.);
    // For the GEBT proof of concept problem (i.e. the clamped beam), the dofs are enforced to be
    // zero at the left end of the beam, so the constraint residual is simply based on the
    // generalized coordinates at the first node
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            // Construct rotation vector from root node rotation quaternion
            auto rotation_vector = openturbine::gen_alpha_solver::rotation_vector_from_quaternion(
                openturbine::gen_alpha_solver::Quaternion(
                    gen_coords(0, 3), gen_coords(0, 4), gen_coords(0, 5), gen_coords(0, 6)
                )
            );
            // Set residual as translation and rotation of root node
            // TODO: update when position & rotations are prescribed
            constraints_residual(0) = gen_coords(0, 0);
            constraints_residual(1) = gen_coords(0, 1);
            constraints_residual(2) = gen_coords(0, 2);
            constraints_residual(3) = rotation_vector.GetXComponent();
            constraints_residual(4) = rotation_vector.GetYComponent();
            constraints_residual(5) = rotation_vector.GetZComponent();
        }
    );
}

}  // namespace openturbine::gebt_poc