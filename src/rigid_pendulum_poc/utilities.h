#pragma once

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
 * @brief  Creates an identity vector of size (size x 1)
 * @param  size: Size of the identity vector
 */
HostView1D create_identity_vector(size_t size);

/*!
 * @brief  Creates an identity matrix of size (size x size)
 * @param  size: Size of the identity matrix
 */
HostView2D create_identity_matrix(size_t size);

}  // namespace openturbine::rigid_pendulum
