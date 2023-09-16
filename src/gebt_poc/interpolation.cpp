#include "src/gebt_poc/interpolation.h"

namespace openturbine::gebt_poc {

Point FindNearestNeighbor(const std::vector<Point>& points, const Point& point) {
    if (points.empty()) {
        throw std::invalid_argument("Points list must not be empty");
    }

    auto nearest_neighbor =
        std::min_element(points.begin(), points.end(), [&point](const Point& lhs, const Point& rhs) {
            return point.DistanceTo(lhs) < point.DistanceTo(rhs);
        });

    return *nearest_neighbor;
}

std::vector<Point> FindkNearestNeighbors(
    const std::vector<Point>& points, const Point& point, const size_t n
) {
    std::vector<Point> remaining_pts{};
    std::copy(points.begin(), points.end(), std::back_inserter(remaining_pts));

    std::vector<Point> k_nearest_neighbors{};
    while (k_nearest_neighbors.size() < n && !remaining_pts.empty()) {
        auto nn = FindNearestNeighbor(remaining_pts, point);
        k_nearest_neighbors.emplace_back(nn);

        remaining_pts.erase(
            std::remove_if(
                remaining_pts.begin(), remaining_pts.end(),
                [&nn](const Point& p) {
                    return p == nn;
                }
            ),
            remaining_pts.end()
        );
    }
    return k_nearest_neighbors;
}

}  // namespace openturbine::gebt_poc
