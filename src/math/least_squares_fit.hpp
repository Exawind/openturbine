#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include <KokkosLapack_gesv.hpp>
#include <Kokkos_Core.hpp>

#include "src/beams/interpolation.hpp"

namespace openturbine {

/**
 * @brief Maps input geometric locations -> normalized domain using linear mapping
 *
 * @param geom_locations Input geometric locations (typically in domain [0, 1]),
 *                       sorted in ascending order
 * @return std::vector<double> Mapped/normalized evaluation points in domain [-1, 1]
 */
inline std::vector<double> MapGeometricLocations(const std::vector<double>& geom_locations) {
    // Get first and last points of the input domain (assumed to be sorted)
    const double domain_start = geom_locations.front();
    const double domain_end = geom_locations.back();
    if (domain_end == domain_start) {
        throw std::invalid_argument(
            "Invalid geometric locations: domain start and end points are equal."
        );
    }

    // Map each point from domain -> [-1, 1]
    std::vector<double> mapped_locations(geom_locations.size());
    const auto domain_span = domain_end - domain_start;
    for (size_t i = 0; i < geom_locations.size(); ++i) {
        mapped_locations[i] = 2. * (geom_locations[i] - domain_start) / domain_span - 1.;
    }
    return mapped_locations;
}

/**
 * @brief Computes shape function matrices ϕg at points ξg
 *
 * @param n Number of geometric points to fit (>=2)
 * @param p Number of points representing the polynomial of order p-1 (2 <= p <= n)
 * @param evaluation_pts Evaluation points in [-1, 1]
 * @return Tuple containing shape function matrix and GLL points
 */
inline std::tuple<std::vector<std::vector<double>>, std::vector<double>> ShapeFunctionMatrices(
    size_t n, size_t p, const std::vector<double>& evaluation_pts
) {
    // Compute GLL points which will act as the nodes for the shape functions
    auto gll_pts = GenerateGLLPoints(p - 1);

    // Compute weights for the shape functions at the evaluation points
    std::vector<double> weights(p, 0.);
    std::vector<std::vector<double>> shape_functions(p, std::vector<double>(n, 0.));
    for (size_t j = 0; j < n; ++j) {
        LagrangePolynomialInterpWeights(evaluation_pts[j], gll_pts, weights);
        for (size_t k = 0; k < p; ++k) {
            shape_functions[k][j] = weights[k];
        }
    }

    return {shape_functions, gll_pts};
}

inline std::vector<std::array<double, 3>> SolveLinearSystem(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    const auto A_internal = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>("A", A.size(), A.front().size());
    const auto B_internal = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>("B", B.size(), B.front().size());
    const auto IPIV = Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::HostSpace>("IPIV", B.size());

    for(auto i = 0U; i < A.size(); ++i) {
        for(auto j = 0U; j < A.front().size(); ++j) {
            A_internal(i, j) = A[i][j];
        }
    }

    for(auto i = 0U; i < B.size(); ++i) {
        for(auto j = 0U; j < B.front().size(); ++j) {
            B_internal(i, j) = B[i][j];
        }
    }

    KokkosLapack::gesv(A_internal, B_internal, IPIV);

    auto result = std::vector<std::array<double, 3>>(B.size());

    for(auto i = 0U; i < result.size(); ++i) {
        for(auto j = 0U; j < result.front().size(); ++j) {
            result[i][j] = B_internal(i, j);
        }
    }

    return result;
}
/**
 * @brief Performs least squares fitting to determine interpolation coefficients
 * @details Performs least squares fitting to determine interpolation coefficients
 *          by solving a dense linear system [A][X] = [B], where [A] is the shape
 *          function matrix (p x n), [B] is the input points (n x 3), and [X] is the
 *          interpolation coefficients (p x 3)
 *
 * @param p Number of points representing the polynomial of order p-1
 * @param shape_functions Shape function matrix (p x n)
 * @param points_to_fit x,y,z coordinates of the points to fit (n x 3)
 * @return Interpolation coefficients (p x 3)
 */
inline std::vector<std::array<double, 3>> PerformLeastSquaresFitting(
    size_t p, const std::vector<std::vector<double>>& shape_functions,
    const std::vector<std::array<double, 3>>& points_to_fit
) {
    if (shape_functions.size() != p) {
        throw std::invalid_argument("shape_functions rows do not match order p.");
    }
    const size_t n = shape_functions[0].size();
    for (const auto& row : shape_functions) {
        if (row.size() != n) {
            throw std::invalid_argument("Inconsistent number of columns in shape_functions.");
        }
    }

    // Construct matrix A in LHS (p x p)
    std::vector<std::vector<double>> A(p, std::vector<double>(p, 0.));
    A[0][0] = 1.;
    A[p - 1][p - 1] = 1.;
    for (size_t i = 1; i < p - 1; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.;
            for (size_t k = 0; k < n; ++k) {
                sum += shape_functions[i][k] * shape_functions[j][k];
            }
            A[i][j] = sum;
        }
    }

    // Construct matrix B in RHS (p x 3)
    std::vector<std::vector<double>> B(p, std::vector<double>(3, 0.));
    for (size_t dim = 0; dim < 3; ++dim) {
        B[0][dim] = points_to_fit[0][dim];
        B[p - 1][dim] = points_to_fit[n - 1][dim];
    }
    for (size_t i = 1; i < p - 1; ++i) {
        for (size_t k = 0; k < n; ++k) {
            for (size_t dim = 0; dim < 3; ++dim) {
                B[i][dim] += shape_functions[i][k] * points_to_fit[k][dim];
            }
        }
    }

    // Solve the least squares problem for each dimension of B
    return SolveLinearSystem(A, B);
}

}  // namespace openturbine
