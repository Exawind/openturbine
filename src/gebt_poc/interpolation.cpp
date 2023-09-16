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

Kokkos::View<double**> LinearInterpolation(
    const Kokkos::View<double**> m1, const Kokkos::View<double**> m2, const double alpha
) {
    if (m1.extent(0) != m2.extent(0) || m1.extent(1) != m2.extent(1)) {
        throw std::invalid_argument("Matrices must have the same dimensions");
    }

    Kokkos::View<double**> interpolated_matrix("interpolated_matrix", m1.extent(0), m1.extent(1));
    for (size_t i = 0; i < m1.extent(0); i++) {
        for (size_t j = 0; j < m1.extent(1); j++) {
            interpolated_matrix(i, j) = (1. - alpha) * m1(i, j) + alpha * m2(i, j);
        }
    }

    return interpolated_matrix;
}

}  // namespace openturbine::gebt_poc
