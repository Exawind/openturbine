#include "src/gebt_poc/solver.h"

namespace openturbine::gebt_poc {

UserDefinedQuadratureRule::UserDefinedQuadratureRule(
    std::vector<double> quadrature_points, std::vector<double> quadrature_weights
)
    : quadrature_points_(std::move(quadrature_points)),
      quadrature_weights_(std::move(quadrature_weights)) {
}

}  // namespace openturbine::gebt_poc
