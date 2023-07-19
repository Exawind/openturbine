#pragma once

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
bool close_to(double a, double b, double epsilon = kTOLERANCE);

/*!
 * @brief  Takes an angle and returns the equivalent angle in the range [-pi, pi]
 * @param  angle: Angle to be wrapped, in radians
 */
double wrap_angle_to_pi(double angle);

}  // namespace openturbine::rigid_pendulum
