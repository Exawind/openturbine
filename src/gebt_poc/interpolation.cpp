#include "src/gebt_poc/interpolation.h"

namespace openturbine::gebt_poc {

Point FindNearestNeighbor(const std::vector<Point>& points, const Point& pt) {
    if (points.empty()) {
        throw std::invalid_argument("Points list must not be empty");
    }

    // Find the nearest neighbor using std algorithm
    auto nearest_neighbor =
        std::min_element(points.begin(), points.end(), [&pt](const Point& lhs, const Point& rhs) {
            return pt.DistanceTo(lhs) < pt.DistanceTo(rhs);
        });

    return *nearest_neighbor;
}

}  // namespace openturbine::gebt_poc
