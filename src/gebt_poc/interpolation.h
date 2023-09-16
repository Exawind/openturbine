#pragma once

#include "src/gebt_poc/point.h"

namespace openturbine::gebt_poc {

/// Find the nearest neighbor of a point from a list
Point FindNearestNeighbor(const std::vector<Point>&, const Point&);

/// Find k-nearest neighbors of a point from a list
std::vector<Point> FindkNearestNeighbors(const std::vector<Point>&, const Point&, const size_t);

}  // namespace openturbine::gebt_poc
