#pragma once

#include "src/gebt_poc/interpolation.h"
#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

// Calculates the Jacobian of transformation for 1-D element based on the shape function derivatives
// and the nodal coordinates of the element in the global csys
// Physically, the Jacobian represents the following (In 1-D):
// Length of element in the global csys / Length of element in the natural csys
double CalculateJacobian(VectorFieldView::const_type nodes, View1D::const_type shape_derivatives);

}  // namespace openturbine::gebt_poc
