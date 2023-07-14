#include "src/rigid_pendulum_poc/quaternion.h"

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

Quaternion quaternion_from_rotation_vector(const Vector& vector) {
    auto [v0, v1, v2] = vector;
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

Vector rotation_vector_from_quaternion(const Quaternion& quaternion) {
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

Quaternion quaternion_from_axis_angle(double angle, const Vector& axis) {
    auto [v0, v1, v2] = axis;
    double sin_angle = std::sin(angle / 2.0);
    double cos_angle = std::cos(angle / 2.0);

    return Quaternion(cos_angle, v0 * sin_angle, v1 * sin_angle, v2 * sin_angle);
}

std::tuple<double, Vector> axis_angle_from_quaternion(const Quaternion& quaternion) {
    auto [q0, q1, q2, q3] = quaternion.GetComponents();
    double angle = 2. * std::atan2(std::sqrt(q1 * q1 + q2 * q2 + q3 * q3), q0);

    // If angle is null, return the angle 0 and axis {1, 0, 0}
    if (close_to(angle, 0.)) {
        return {0., {1., 0., 0.}};
    }

    angle = wrap_angle_to_pi(angle);
    double k = 1. / std::sqrt(q1 * q1 + q2 * q2 + q3 * q3);

    // normalize the axis
    // TODO: Take care of the tech debt (Vector class) and use its normalize method
    auto axis = Vector{q1 * k, q2 * k, q3 * k};
    auto [v0, v1, v2] = axis;
    auto length = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    auto normalized_axis = Vector{v0 / length, v1 / length, v2 / length};

    return {angle, normalized_axis};
}

Vector rotate_vector(const Quaternion& quaternion, const Vector& vector) {
    auto [v0, v1, v2] = vector;
    auto [q0, q1, q2, q3] = quaternion.GetComponents();

    return Vector(
        (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * v0 + 2. * (q1 * q2 - q0 * q3) * v1 +
            2. * (q1 * q3 + q0 * q2) * v2,
        2. * (q1 * q2 + q0 * q3) * v0 + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * v1 +
            2. * (q2 * q3 - q0 * q1) * v2,
        2. * (q1 * q3 - q0 * q2) * v0 + 2. * (q2 * q3 + q0 * q1) * v1 +
            (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * v2
    );
}

}  // namespace openturbine::rigid_pendulum
