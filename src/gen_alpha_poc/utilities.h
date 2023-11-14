#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

namespace openturbine::gen_alpha_solver {

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
KOKKOS_INLINE_FUNCTION
double wrap_angle_to_pi(double angle) {
    double wrapped_angle = std::fmod(angle, 2. * kPi);

    if (close_to(wrapped_angle, kPi)) {
        return kPi;
    }
    else if (close_to(wrapped_angle, -kPi)) {
        return -kPi;
    }
    else if (wrapped_angle > kPi) {
        return wrapped_angle - 2. * kPi;
    }
    else if (wrapped_angle < -kPi) {
        return wrapped_angle + 2. * kPi;
    }
    return wrapped_angle;
}

/// Creates an identity vector (i.e. a vector with all entries equal to 1)
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void create_identity_vector(Kokkos::View<double*, MemorySpace> vector) {
  for(std::size_t i = 0; i < vector.extent(0); ++i) {
    vector(i) = 0.;
  }
}

/// Creates an identity matrix (i.e. a diagonal matrix with all diagonal entries equal to 1)
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void create_identity_matrix(Kokkos::View<double**, MemorySpace> matrix) {
    for(std::size_t i = 0; i < matrix.extent(0); ++i) {
        for(std::size_t j = 0; j < matrix.extent(1); ++j) {
            matrix(i, j) = (i == j);
        }
    }
}

/// Returns an n x 1 vector with provided values from a vector
Kokkos::View<double*> create_vector(const std::vector<double>&);

/// Creates a m x n matrix with provided values from a 2D vector
Kokkos::View<double**> create_matrix(const std::vector<std::vector<double>>&);

/// Transposes a provided m x n matrix and returns an n x m matrix
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void transpose_matrix(Kokkos::View<const double**, MemorySpace> matrix, Kokkos::View<double**, MemorySpace> transposed_matrix) {
    for(std::size_t row = 0; row < matrix.extent(0); ++row) {
        for(std::size_t column = 0; column < matrix.extent(0); ++column) {
            transposed_matrix(row, column) = matrix(column, row);
        }
    }
}

/// Generates and returns the 3 x 3 cross product matrix from a provided 3D vector
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void create_cross_product_matrix(Kokkos::View<const double[3], MemorySpace> vector, Kokkos::View<double[3][3], MemorySpace> matrix) {
    matrix(0, 0) = 0.;
    matrix(0, 1) = -vector(2);
    matrix(0, 2) = vector(1);
    matrix(1, 0) = vector(2);
    matrix(1, 1) = 0.;
    matrix(1, 2) = -vector(0);
    matrix(2, 0) = -vector(1);
    matrix(2, 1) = vector(0);
    matrix(2, 2) = 0.;
}

/// Multiplies an n x 1 vector with a scalar and returns an n x 1 vector
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void multiply_vector_with_scalar(Kokkos::View<const double*, MemorySpace> vector, double scalar, Kokkos::View<double*, MemorySpace> result) {
  for(std::size_t i = 0; i < vector.extent(0); ++i) {
    result(i) = vector * scalar;
  }
}

/// Multiplies an m x n matrix with a scalar and returns an m x n matrix
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void multiply_matrix_with_scalar(Kokkos::View<const double**, MemorySpace> matrix, double scalar, Kokkos::View<double**, MemorySpace> result) {
  for(std::size_t row = 0; row < matrix.extent(0); ++row) {
    for(std::size_t column = 0; column < matrix.extent(1); ++column) {
        result(row, column) = matrix(row, column) * scalar;
    }
  }
}

/// Multiplies an 1 x n vector with an n x 1 vector and returns a double
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
double dot_product(Kokkos::View<const double*, MemorySpace> vector_1, Kokkos::View<const double*, MemorySpace> vector_2) {
    double result = 0.;
    for(std::size_t i = 0; i < vector_1.extent(0); ++i) {
        result += vector_1(i) * vector_2(i);
    }
    return result;
}

// /// Multiplies an m x n matrix with an n x 1 vector and returns an m x 1 vector
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void multiply_matrix_with_vector(Kokkos::View<const double**, MemorySpace> matrix, Kokkos::View<const double*, MemorySpace> vector, Kokkos::View<double*, MemorySpace> result) {
    for(std::size_t i = 0; i < result.extent(0); ++i) {
        result(i) = dot_product(Kokkos::subview(matrix, i, Kokkos::ALL), vector);
    }
}

/// Multiplies an m x n matrix with an n x p matrix and returns an m x p matrix
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void multiply_matrix_with_matrix(Kokkos::View<const double**, MemorySpace> matrix_1, Kokkos::View<const double**, MemorySpace> matrix_2, Kokkos::View<double**, MemorySpace> result) {
    for(std::size_t i = 0; i < result.extent(0); ++i) {
        for(std::size_t j = 0; j < result.extent(1); ++j) {
            result(i, j) = dot_product(Kokkos::subview(matrix_1, i, Kokkos::ALL), Kokkos::subview(matrix_2, Kokkos::ALL, j));
        }
    }
}

/// Adds an m x n matrix with an m x n matrix and returns an m x n matrix
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void add_matrix_with_matrix(Kokkos::View<const double**, MemorySpace> matrix_1, Kokkos::View<const double**, MemorySpace> matrix_2, Kokkos::View<double**, MemorySpace> result) {
    for(std::size_t i = 0; i < result.extent(0); ++i) {
        for(std::size_t j = 0; j < result.extent(1); ++j) {
            result(i, j) = matrix_1(i, j) + matrix_2(i, j);
        }
    }
}

}  // namespace openturbine::gen_alpha_solver
