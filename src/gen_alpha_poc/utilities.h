#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

#include "src/utilities/openturbine_types.h"

namespace openturbine::gen_alpha_solver {

using openturbine::util::HostView1D;
using openturbine::util::HostView2D;

static constexpr double kTolerance = 1e-6;
static constexpr double kPi = 3.14159265358979323846;

/*!
 * @brief  Returns a boolean indicating if two provided doubles are close to each other
 * @param  a: First double
 * @param  b: Second double
 * @param  epsilon: Tolerance for closeness
 */
bool close_to(double a, double b, double epsilon = kTolerance);

/*!
 * @brief  Takes an angle and returns the equivalent angle in the range [-pi, pi]
 * @param  angle: Angle to be wrapped, in radians
 */
double wrap_angle_to_pi(double angle);

/// Creates an identity vector (i.e. a vector with all entries equal to 1)
HostView1D create_identity_vector(size_t size);

/// Creates an identity matrix (i.e. a diagonal matrix with all diagonal entries equal to 1)
HostView2D create_identity_matrix(size_t size);

/// Returns an n x 1 vector with provided values from a vector
HostView1D create_vector(const std::vector<double>&);

/// Creates a m x n matrix with provided values from a 2D vector
HostView2D create_matrix(const std::vector<std::vector<double>>&);

/// Transposes a provided m x n matrix and returns an n x m matrix
HostView2D transpose_matrix(const HostView2D);

/// Generates and returns the 3 x 3 cross product matrix from a provided 3D vector
HostView2D create_cross_product_matrix(const HostView1D);

/// Multiplies an m x n matrix with a scalar and returns an m x n matrix
HostView2D multiply_matrix_with_scalar(const HostView2D, double);

/// Multiplies an m x n matrix with an n x 1 vector and returns an m x 1 vector
HostView1D multiply_matrix_with_vector(const HostView2D, const HostView1D);

/// Multiplies an m x n matrix with an n x p matrix and returns an m x p matrix
HostView2D multiply_matrix_with_matrix(const HostView2D, const HostView2D);

}  // namespace openturbine::gen_alpha_solver
