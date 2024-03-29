#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

namespace openturbine::gen_alpha_solver {

static constexpr double kEpsilon{std::numeric_limits<double>::epsilon()};
static constexpr double kTolerance = 1e-6;
static constexpr double kPi = 3.14159265358979323846;

/*!
 * @brief  Returns a boolean indicating if two provided doubles are close to each other
 * @param  a: First double
 * @param  b: Second double
 * @param  epsilon: Tolerance for closeness
 */
KOKKOS_INLINE_FUNCTION
bool close_to(double a, double b, double epsilon = kTolerance) {
    auto delta = std::abs(a - b);
    a = std::abs(a);
    b = std::abs(b);

    if (a < epsilon) {
        if (b < epsilon) {
            return true;
        }
        return false;
    }

    return (delta / a) < epsilon ? true : false;
}

/*!
 * @brief  Takes an angle and returns the equivalent angle in the range [-pi, pi]
 * @param  angle: Angle to be wrapped, in radians
 */
double wrap_angle_to_pi(double angle);

/// Creates an identity vector (i.e. a vector with all entries equal to 1)
Kokkos::View<double*> create_identity_vector(size_t size);

/// Creates an identity matrix (i.e. a diagonal matrix with all diagonal entries equal to 1)
Kokkos::View<double**> create_identity_matrix(size_t size);

/// Returns an n x 1 vector with provided values from a vector
Kokkos::View<double*> create_vector(const std::vector<double>&);

/// Creates a m x n matrix with provided values from a 2D vector
Kokkos::View<double**> create_matrix(const std::vector<std::vector<double>>&);

/// Transposes a provided m x n matrix and returns an n x m matrix
Kokkos::View<double**> transpose_matrix(const Kokkos::View<double**>);

/// Generates and returns the 3 x 3 cross product matrix from a provided 3D vector
template <typename VectorType>
inline Kokkos::View<double**> create_cross_product_matrix(VectorType vector) {
    if (vector.extent(0) != 3) {
        throw std::invalid_argument("The provided vector must have 3 elements");
    }

    auto matrix = Kokkos::View<double[3][3]>("cross_product_matrix");
    auto populate_matrix = KOKKOS_LAMBDA(size_t) {
        matrix(0, 0) = 0.;
        matrix(0, 1) = -vector(2);
        matrix(0, 2) = vector(1);
        matrix(1, 0) = vector(2);
        matrix(1, 1) = 0.;
        matrix(1, 2) = -vector(0);
        matrix(2, 0) = -vector(1);
        matrix(2, 1) = vector(0);
        matrix(2, 2) = 0.;
    };

    Kokkos::parallel_for(1, populate_matrix);

    return matrix;
}

/// Convert a 3x3 cross product matrix to a 3D vector
Kokkos::View<double*> convert_cross_product_matrix_to_vector(Kokkos::View<double[3][3]>);

/// Multiplies an n x 1 vector with a scalar and returns an n x 1 vector
Kokkos::View<double*> multiply_vector_with_scalar(const Kokkos::View<double*>, double);

/// Multiplies an m x n matrix with a scalar and returns an m x n matrix
Kokkos::View<double**> multiply_matrix_with_scalar(const Kokkos::View<double**>, double);

/// Multiplies an m x n matrix with an n x 1 vector and returns an m x 1 vector
Kokkos::View<double*>
multiply_matrix_with_vector(const Kokkos::View<double**>, const Kokkos::View<double*>);

Kokkos::View<double*>
multiply_matrix_with_vector(const Kokkos::View<const double**>, const Kokkos::View<double*>);

/// Multiplies an m x n matrix with an n x p matrix and returns an m x p matrix
Kokkos::View<double**>
multiply_matrix_with_matrix(const Kokkos::View<double**>, const Kokkos::View<double**>);

Kokkos::View<double**>
multiply_matrix_with_matrix(const Kokkos::View<const double**>, const Kokkos::View<const double**>);

/// Adds an m x n matrix with an m x n matrix and returns an m x n matrix
Kokkos::View<double**>
add_matrix_with_matrix(const Kokkos::View<double**>, const Kokkos::View<double**>);

Kokkos::View<double**>
add_matrix_with_matrix(const Kokkos::View<const double**>, const Kokkos::View<const double**>);

}  // namespace openturbine::gen_alpha_solver
