#pragma once

#include "src/gebt_poc/interpolation.h"

namespace openturbine::gebt_poc {

// Calculates the Jacobian of transformation for 1-D element based on the shape function derivatives
// and the nodal coordinates of the element in the global csys
// Physically, the Jacobian represents the following (In 1-D):
// Length of element in the global csys / Length of element in the natural csys
double CalculateJacobian(Kokkos::View<const double*[3]> nodes, Kokkos::View<const double*> shape_derivatives);

}  // namespace openturbine::gebt_poc
