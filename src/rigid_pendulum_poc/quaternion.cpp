#include "src/rigid_pendulum_poc/quaternion.h"

#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

bool close_to(double a, double b) {
    auto delta = std::abs(a - b);
    a = std::abs(a);
    b = std::abs(b);

    if (a < 1e-6) {
        if (b < 1e-6) {
            return true;
        }
        return false;
    }

    return (delta / a) < kTOLERANCE ? true : false;
}

Quaternion::Quaternion(double q0, double q1, double q2, double q3)
    : q0_(q0), q1_(q1), q2_(q2), q3_(q3) {
}

bool Quaternion::IsUnitQuaternion() const {
    return close_to(Length(), 1.);
}

Quaternion Quaternion::GetUnitQuaternion() const {
    double length = Length();

    // Return the quaternion itself if unit or null quaternion
    if (close_to(length, 0.) || close_to(length, 1.)) {
        return *this;
    }

    return *this / length;
}

Quaternion quaternion_from_rotation_vector(const std::tuple<double, double, double>& vector) {
    auto v0 = std::get<0>(vector);
    auto v1 = std::get<1>(vector);
    auto v2 = std::get<2>(vector);

    double angle = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);

    // Return the quaternion {1, 0, 0, 0} if provided rotation vector is null
    if (close_to(angle, 0.)) {
        return Quaternion(1.0, 0.0, 0.0, 0.0);
    }

    double sin_angle = std::sin(angle / 2.0);
    double cos_angle = std::cos(angle / 2.0);
    auto factor = sin_angle / angle;

    return Quaternion(cos_angle, v0 * factor, v1 * factor, v2 * factor);
}

std::tuple<double, double, double> rotation_vector_from_quaternion(const Quaternion& quaternion) {
    auto [q0, q1, q2, q3] = quaternion.GetComponents();
    auto sin_angle_squared = q1 * q1 + q2 * q2 + q3 * q3;

    // Return the rotation vector {0, 0, 0} if provided quaternion is null
    if (close_to(sin_angle_squared, 0.)) {
        return {0., 0., 0.};
    }

    double sin_angle = std::sqrt(sin_angle_squared);
    double k = 2. * std::atan2(sin_angle, q0) / sin_angle;

    return {q1 * k, q2 * k, q3 * k};
}

}  // namespace openturbine::rigid_pendulum
