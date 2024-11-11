#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "KokkosLapack_gesv.hpp"

#include "src/beams/interpolation.hpp"

namespace openturbine {

// Step 1: Mapping Geometric Locations
/**
 * @brief Maps input geometric locations -> normalized domain using linear mapping
 *
 * @param geom_input_locations Input geometric locations (typically in domain [0, 1]),
 *                             sorted
 * @return std::vector<double> Mapped/normalized evaluation points in domain [-1, 1]
 */
std::vector<double> MapGeometricLocations(const std::vector<double>& geom_input_locations) {
    // Get first and last points of the input domain (assumed to be sorted)
    double domain_start = geom_input_locations.front();
    double domain_end = geom_input_locations.back();
    if (domain_end == domain_start) {
        throw std::invalid_argument(
            "Invalid geometric locations: domain start and end points are equal."
        );
    }

    // Map each point from domain -> [-1, 1]
    std::vector<double> mapped_locations(geom_input_locations.size());
    auto domain_span = domain_end - domain_start;
    for (size_t i = 0; i < geom_input_locations.size(); ++i) {
        mapped_locations[i] = 2. * (geom_input_locations[i] - domain_start) / domain_span - 1.;
    }
    return mapped_locations;
}

// Step 2: Shape Function Matrices
/**
 * @brief Computes shape function matrices ϕg at points ξg
 *
 * @param n Number of geometric points to fit (>=2)
 * @param p Number of points representing the polynomial of order p-1 (2 <= p <= n)
 * @param xi_g Evaluation points in [-1, 1]
 * @return Tuple containing shape function matrix and GLL points
 */
std::tuple<std::vector<std::vector<double>>, std::vector<double>> ShapeFunctionMatrices(
    size_t n, size_t p, const std::vector<double>& xi_g
) {
    // Compute GLL points which will act as the nodes for the shape functions
    auto gll_pts = GenerateGLLPoints(p - 1);

    // Compute weights for the shape functions at the nodes
    std::vector<double> weights(p, 0.);
    std::vector<std::vector<double>> phi_g(p, std::vector<double>(n, 0.));
    for (size_t j = 0; j < n; ++j) {
        LagrangePolynomialInterpWeights(xi_g[j], gll_pts, weights);
        for (size_t k = 0; k < p; ++k) {
            phi_g[k][j] = weights[k];
        }
    }

    return {phi_g, gll_pts};
}

/**
 * @brief Solves a linear system Ax = b
 * @param A Square coefficient matrix (will be modified during solving)
 * @param b Right-hand side vector (will be modified during solving)
 * @return Solution vector x
 */
std::vector<double> SolveLinearSystem(
    [[maybe_unused]] std::vector<std::vector<double>>& A, [[maybe_unused]] std::vector<double>& b
) {
    // TODO: Implement linear system solver
    return std::vector<double>(b.size(), 0.);
}

// Step 3: Least Squares Fitting
/**
 * @brief Performs least squares fitting to determine interpolation coefficients X
 *
 * @param p Number of points representing the polynomial of order p-1
 * @param phi_g Shape function matrix (p x n)
 * @param Xk Input geometric data (n x 3)
 * @return Interpolation coefficients X (p x 3)
 */
std::vector<std::array<double, 3>> PerformLeastSquaresFitting(
    size_t p, const std::vector<std::vector<double>>& phi_g,
    const std::vector<std::array<double, 3>>& Xk
) {
    if (phi_g.size() != p) {
        throw std::invalid_argument("phi_g rows do not match order p.");
    }
    size_t n = phi_g[0].size();
    for (const auto& row : phi_g) {
        if (row.size() != n) {
            throw std::invalid_argument("Inconsistent number of columns in phi_g.");
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
                sum += phi_g[i][k] * phi_g[j][k];
            }
            A[i][j] = sum;
        }
    }

    // Construct matrix B in RHS (p x 3)
    std::vector<std::vector<double>> B(p, std::vector<double>(3, 0.));
    for (size_t dim = 0; dim < 3; ++dim) {
        B[0][dim] = Xk[0][dim];
        B[p - 1][dim] = Xk[n - 1][dim];
    }
    for (size_t i = 1; i < p - 1; ++i) {
        for (size_t k = 0; k < n; ++k) {
            for (size_t dim = 0; dim < 3; ++dim) {
                B[i][dim] += phi_g[i][k] * Xk[k][dim];
            }
        }
    }

    // Solve the least squares problem for each dimension of B
    std::vector<std::array<double, 3>> X_coefficients(p, {0., 0., 0.});
    for (size_t dim = 0; dim < 3; ++dim) {
        std::vector<double> B_col(p, 0.);
        for (size_t i = 0; i < p; ++i) {
            B_col[i] = B[i][dim];
        }

        auto X = SolveLinearSystem(A, B_col);
        for (size_t i = 0; i < p; ++i) {
            X_coefficients[i][dim] = X[i];
        }
    }
    return X_coefficients;
}

}  // namespace openturbine
