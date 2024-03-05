#pragma once

#include "src/gebt_poc/element.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {
/// Calculates the constraint gradient matrix for a beam element
void ElementalConstraintForcesGradientMatrix(
    View1D::const_type gen_coords, View1D::const_type position_vector,
    View2D constraints_gradient_matrix
);

}  // namespace openturbine::gebt_poc
