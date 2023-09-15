#pragma once

#include "src/gebt_poc/point.h"

namespace openturbine::gebt_poc {

/// Find the nearest neighbor of a point from a list
Point FindNearestNeighbor(const std::vector<Point>&, const Point&);

}  // namespace openturbine::gebt_poc
