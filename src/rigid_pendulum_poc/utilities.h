#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

// TODO: Move the following definitions to a constants.h file in a common math directory
static constexpr double kTOLERANCE = 1e-6;
static constexpr double kPI = 3.14159265358979323846;

// TODO: Move the following math related functions to a common math directory
/*!
 * @brief  Returns a boolean indicating if two provided doubles are close to each other
 * @param  a: First double
 * @param  b: Second double
 * @param  epsilon: Tolerance for closeness
 */
KOKKOS_FUNCTION
bool close_to(double a, double b, double epsilon = kTOLERANCE);

/*!
 * @brief  Takes an angle and returns the equivalent angle in the range [-pi, pi]
 * @param  angle: Angle to be wrapped, in radians
 */
double wrap_angle_to_pi(double angle);

/*!
 * @brief  Creates an identity vector (i.e. a vector with all entries equal to 1)
 * @param  size: Size of the identity vector
 */
Kokkos::View<double*> create_identity_vector(size_t size);

/*!
 * @brief  Creates an identity matrix (i.e. a diagonal matrix with all diagonal entries equal to 1)
 * @param  size: Size of the identity matrix
 */
Kokkos::View<double**> create_identity_matrix(size_t size);

/*!
 * @brief  Returns a HostView1D with provided values from a vector
 * @param  vector: Values to be used in the vector
 */
Kokkos::View<double*> create_vector(const std::vector<double>&);

/*!
 * @brief  Creates a HostView2D with provided values from a 2D vector
 * @param  vector: Values to be used in the matrix
 */
Kokkos::View<double**> create_matrix(const std::vector<std::vector<double>>&);

/// Transposes a provided m x n matrix and returns an n x m matrix
Kokkos::View<double**> transpose_matrix(const Kokkos::View<double**>);

/// Generates and returns the 3 x 3 cross product matrix from a provided 3D vector
Kokkos::View<double**> create_cross_product_matrix(const Kokkos::View<double*>);

/// Multiplies an m x n matrix with an n x 1 vector and returns an m x 1 vector
Kokkos::View<double*> multiply_matrix_with_vector(const Kokkos::View<double**>, const Kokkos::View<double*>);

/// Multiplies an m x n matrix with an n x p matrix and returns an m x p matrix
Kokkos::View<double**> multiply_matrix_with_matrix(const Kokkos::View<double**>, const Kokkos::View<double**>);

/// Multiplies an m x n matrix with a scalar and returns an m x n matrix
Kokkos::View<double**> multiply_matrix_with_scalar(const Kokkos::View<double**>, double);

}  // namespace openturbine::rigid_pendulum
