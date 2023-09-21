#include "src/gebt_poc/interpolation.h"

namespace openturbine::gebt_poc {

Point FindNearestNeighbor(const std::vector<Point>& points_list, const Point& point) {
    if (points_list.empty()) {
        throw std::invalid_argument("points_list list must not be empty");
    }

    auto nearest_neighbor = std::min_element(
        points_list.begin(), points_list.end(),
        [&point](const Point& lhs, const Point& rhs) {
            return point.DistanceTo(lhs) < point.DistanceTo(rhs);
        }
    );

    return *nearest_neighbor;
}

std::vector<Point> FindkNearestNeighbors(
    const std::vector<Point>& points_list, const Point& point, const size_t k
) {
    std::vector<Point> remaining_pts{};
    std::copy(points_list.begin(), points_list.end(), std::back_inserter(remaining_pts));

    std::vector<Point> k_nearest_neighbors{};
    while (k_nearest_neighbors.size() < k && !remaining_pts.empty()) {
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

Kokkos::View<double**> LinearlyInterpolateMatrices(
    const Kokkos::View<double**> m1, const Kokkos::View<double**> m2, const double alpha
) {
    if (m1.extent(0) != m2.extent(0) || m1.extent(1) != m2.extent(1)) {
        throw std::invalid_argument("Matrices must have the same dimensions");
    }

    Kokkos::View<double**> interpolated_matrix("interpolated_matrix", m1.extent(0), m1.extent(1));
    auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
        {0, 0}, {m1.extent(0), m1.extent(1)}
    );
    auto interpolate_row_column = KOKKOS_LAMBDA(size_t row, size_t column) {
        interpolated_matrix(row, column) = (1. - alpha) * m1(row, column) + alpha * m2(row, column);
    };
    Kokkos::parallel_for(entries, interpolate_row_column);

    return interpolated_matrix;
}

}  // namespace openturbine::gebt_poc
