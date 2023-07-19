#include "src/rigid_pendulum_poc/utilities.h"

#include <cmath>

namespace openturbine::rigid_pendulum {

bool close_to(double a, double b, double epsilon) {
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

double wrap_angle_to_pi(double angle) {
    double wrapped_angle = std::fmod(angle, 2. * kPI);

    // Check if the angle is close to PI or -PI to avoid numerical issues
    if (close_to(wrapped_angle, kPI)) {
        return kPI;
    }
    if (close_to(wrapped_angle, -kPI)) {
        return -kPI;
    }

    if (wrapped_angle > kPI) {
        wrapped_angle -= 2. * kPI;
    }
    if (wrapped_angle < -kPI) {
        wrapped_angle += 2. * kPI;
    }

    return wrapped_angle;
}

}  // namespace openturbine::rigid_pendulum
