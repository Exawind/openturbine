#include "src/gebt_poc/interpolation.h"

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

Point FindNearestNeighbor(const std::vector<Point>& points_list, const Point& point) {
    if (points_list.empty()) {
        throw std::invalid_argument("points list must not be empty");
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

double LegendrePolynomial(const size_t n, const double x) {
    if (n == 0) {
        return 1.;
    }
    if (n == 1) {
        return x;
    }
    return (
        ((2 * n - 1) * x * LegendrePolynomial(n - 1, x) - (n - 1) * LegendrePolynomial(n - 2, x)) / n
    );
}

double LegendrePolynomialDerivative(const size_t n, const double x) {
    if (n == 0) {
        return 0.;
    }
    if (n == 1) {
        return 1.;
    }
    if (n == 2) {
        return (3. * x);
    }
    return ((2 * n - 1) * LegendrePolynomial(n - 1, x) + LegendrePolynomialDerivative(n - 2, x));
}

std::vector<double> GenerateGLLPoints(const size_t order) {
    if (order < 1) {
        throw std::invalid_argument("Polynomial order must be greater than or equal to 1");
    }

    auto n_nodes = order + 1;  // number of nodes = order + 1
    std::vector<double> gll_points(n_nodes);
    gll_points[0] = -1.;     // left end point
    gll_points[order] = 1.;  // right end point

    for (size_t i = 1; i < n_nodes; ++i) {
        // Use the Chebyshev-Gauss-Lobatto nodes as the initial guess
        auto x_it = -std::cos(static_cast<double>(i) * M_PI / order);

        // Use Newton's method to find the roots of the Legendre polynomial
        for (size_t j = 0; j < kMaxIterations; ++j) {
            auto x_old = x_it;

            auto legendre_poly = std::vector<double>(n_nodes);
            for (size_t k = 0; k < n_nodes; ++k) {
                legendre_poly[k] = LegendrePolynomial(k, x_it);
            }

            x_it -= (x_it * legendre_poly[n_nodes - 1] - legendre_poly[n_nodes - 2]) /
                    (n_nodes * legendre_poly[n_nodes - 1]);

            if (std::abs(x_it - x_old) <= kConvergenceTolerance) {
                break;
            }
        }
        gll_points[i] = x_it;
    }

    return gll_points;
}

std::vector<double> LagrangePolynomial(const size_t n, const double x) {
    auto gll_points = GenerateGLLPoints(n);  // n+1 GLL points for n+1 nodes
    auto lagrange_poly = std::vector<double>(n + 1, 1.);

    for (size_t i = 0; i < n + 1; ++i) {
        if (gen_alpha_solver::close_to(gll_points[i], x)) {
            // If x is close to a GLL point, then Lagrange polynomial is 1.0 at that point
            continue;
        }
        lagrange_poly[i] =
            ((-1. / (n * (n + 1))) * ((1 - x * x) / (x - gll_points[i])) *
             (LegendrePolynomialDerivative(n, x) / LegendrePolynomial(n, gll_points[i])));
    }

    return lagrange_poly;
}

}  // namespace openturbine::gebt_poc
