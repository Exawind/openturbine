#include "src/rigid_pendulum_poc/quaternion.h"

#include <stdexcept>

namespace openturbine::rigid_pendulum {

// KOKKOS_FUNCTION
// Quaternion::Quaternion(double q0, double q1, double q2, double q3)
//     : q0_(q0), q1_(q1), q2_(q2), q3_(q3) {
// }

KOKKOS_FUNCTION
bool Quaternion::IsUnitQuaternion() const {
    return close_to(Length(), 1.);
}

KOKKOS_FUNCTION
Quaternion Quaternion::GetUnitQuaternion() const {
    double length = Length();

    if (close_to(length, 1.)) {
        return *this;
    }

    return *this / length;
}

KOKKOS_FUNCTION
Quaternion quaternion_from_rotation_vector(const Vector& vector) {
    auto v0 = vector.GetXComponent();
    auto v1 = vector.GetYComponent();
    auto v2 = vector.GetZComponent();

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

KOKKOS_FUNCTION
Vector rotation_vector_from_quaternion(const Quaternion& quaternion) {
    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();

    auto sin_angle_squared = q1 * q1 + q2 * q2 + q3 * q3;

    // Return the rotation vector {0, 0, 0} if provided quaternion is null
    if (close_to(sin_angle_squared, 0.)) {
        return Vector{0.0, 0.0, 0.0};
    }

    double sin_angle = std::sqrt(sin_angle_squared);
    double k = 2. * std::atan2(sin_angle, q0) / sin_angle;

    return {q1 * k, q2 * k, q3 * k};
}

KOKKOS_FUNCTION
Quaternion quaternion_from_angle_axis(double angle, const Vector& axis) {
    auto v0 = axis.GetXComponent();
    auto v1 = axis.GetYComponent();
    auto v2 = axis.GetZComponent();

    double sin_angle = std::sin(angle / 2.0);
    double cos_angle = std::cos(angle / 2.0);

    // We should always get a unit quaternion from the following components
    return Quaternion(cos_angle, v0 * sin_angle, v1 * sin_angle, v2 * sin_angle);
}

std::tuple<double, Vector> angle_axis_from_quaternion(const Quaternion& quaternion) {
    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();

    double angle = 2. * std::atan2(std::sqrt(q1 * q1 + q2 * q2 + q3 * q3), q0);

    // If angle is null, return the angle 0 and axis {1, 0, 0}
    if (close_to(angle, 0.)) {
        return {0., {1., 0., 0.}};
    }

    angle = wrap_angle_to_pi(angle);
    double k = 1. / std::sqrt(q1 * q1 + q2 * q2 + q3 * q3);
    auto normalized_axis = Vector{q1 * k, q2 * k, q3 * k}.GetUnitVector();

    return {angle, normalized_axis};
}

KOKKOS_FUNCTION
Vector rotate_vector(const Quaternion& quaternion, const Vector& vector) {
    auto v0 = vector.GetXComponent();
    auto v1 = vector.GetYComponent();
    auto v2 = vector.GetZComponent();

    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();

    return Vector{
        (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * v0 + 2. * (q1 * q2 - q0 * q3) * v1 +
            2. * (q1 * q3 + q0 * q2) * v2,
        2. * (q1 * q2 + q0 * q3) * v0 + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * v1 +
            2. * (q2 * q3 - q0 * q1) * v2,
        2. * (q1 * q3 - q0 * q2) * v0 + 2. * (q2 * q3 + q0 * q1) * v1 +
            (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * v2};
}

KOKKOS_FUNCTION
RotationMatrix quaternion_to_rotation_matrix(const Quaternion& quaternion) {
    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();

    return RotationMatrix{
        q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
        2. * (q1 * q2 - q0 * q3),
        2. * (q1 * q3 + q0 * q2),
        2. * (q1 * q2 + q0 * q3),
        q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
        2. * (q2 * q3 - q0 * q1),
        2. * (q1 * q3 - q0 * q2),
        2. * (q2 * q3 + q0 * q1),
        q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3};
}

KOKKOS_FUNCTION
Quaternion rotation_matrix_to_quaternion(const RotationMatrix& rotation_matrix) {
    auto trace = rotation_matrix(0, 0) + rotation_matrix(1, 1) + rotation_matrix(2, 2);

    if (trace > 0) {
        auto s = 0.5 / std::sqrt(trace + 1.0);
        return Quaternion{
            0.25 / s, (rotation_matrix(2, 1) - rotation_matrix(1, 2)) * s,
            (rotation_matrix(0, 2) - rotation_matrix(2, 0)) * s,
            (rotation_matrix(1, 0) - rotation_matrix(0, 1)) * s};
    } else if (rotation_matrix(0, 0) > rotation_matrix(1, 1) && rotation_matrix(0, 0) > rotation_matrix(2, 2)) {
        auto s =
            2.0 *
            std::sqrt(1.0 + rotation_matrix(0, 0) - rotation_matrix(1, 1) - rotation_matrix(2, 2));
        return Quaternion(
            (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / s, 0.25 * s,
            (rotation_matrix(0, 1) + rotation_matrix(1, 0)) / s,
            (rotation_matrix(0, 2) + rotation_matrix(2, 0)) / s
        );
    } else if (rotation_matrix(1, 1) > rotation_matrix(2, 2)) {
        auto s =
            2.0 *
            std::sqrt(1.0 + rotation_matrix(1, 1) - rotation_matrix(0, 0) - rotation_matrix(2, 2));
        return Quaternion(
            (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / s,
            (rotation_matrix(0, 1) + rotation_matrix(1, 0)) / s, 0.25 * s,
            (rotation_matrix(1, 2) + rotation_matrix(2, 1)) / s
        );
    } else {
        auto s =
            2.0 *
            std::sqrt(1.0 + rotation_matrix(2, 2) - rotation_matrix(0, 0) - rotation_matrix(1, 1));
        return Quaternion(
            (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / s,
            (rotation_matrix(0, 2) + rotation_matrix(2, 0)) / s,
            (rotation_matrix(1, 2) + rotation_matrix(2, 1)) / s, 0.25 * s
        );
    }
}

}  // namespace openturbine::rigid_pendulum
