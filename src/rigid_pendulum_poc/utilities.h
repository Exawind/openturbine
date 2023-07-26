#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;
using HostView2D = Kokkos::View<double**, Kokkos::HostSpace>;

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
HostView1D create_identity_vector(size_t size);

/*!
 * @brief  Creates an identity matrix (i.e. a diagonal matrix with all diagonal entries equal to 1)
 * @param  size: Size of the identity matrix
 */
HostView2D create_identity_matrix(size_t size);

/*!
 * @brief  Returns a HostView1D with provided values from a vector
 * @param  vector: Values to be used in the vector
 */
HostView1D create_vector(const std::vector<double>&);

/*!
 * @brief  Creates a HostView2D with provided values from a 2D vector
 * @param  vector: Values to be used in the matrix
 */
HostView2D create_matrix(const std::vector<std::vector<double>>&);

/// Transposes a provided m x n matrix and returns an n x m matrix
HostView2D transpose_matrix(HostView2D);

/// Generates and returns the 3 x 3 cross product matrix from a provided 3D vector
HostView2D create_cross_product_matrix(HostView1D);

/// Multiplies an m x n matrix with an n x 1 vector and returns an m x 1 vector
HostView1D multiply_matrix_with_vector(HostView2D, HostView1D);

/// Multiplies an m x n matrix with an n x p matrix and returns an m x p matrix
HostView2D multiply_matrix_with_matrix(HostView2D, HostView2D);

}  // namespace openturbine::rigid_pendulum
