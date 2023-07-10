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

Quaternion::Quaternion(std::array<double, 4> values) : values_(std::move(values)) {
}

Quaternion::Quaternion(double p0, double p1, double p2, double p3) : values_({p0, p1, p2, p3}) {
}

Quaternion::Quaternion(double scalar, const std::array<double, 3>& vector)
    : values_({scalar, vector[0], vector[1], vector[2]}) {
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

Quaternion quaternion_from_rotation_vector(const std::array<double, 3>& rotation_vector) {
    double angle = std::sqrt(
        rotation_vector[0] * rotation_vector[0] + rotation_vector[1] * rotation_vector[1] +
        rotation_vector[2] * rotation_vector[2]
    );

    // Return the quaternion {1, 0, 0, 0} if provided rotation vector is null
    if (close_to(angle, 0.)) {
        return Quaternion(1.0, 0.0, 0.0, 0.0);
    }

    double sin_angle = std::sin(angle / 2.0);
    double cos_angle = std::cos(angle / 2.0);
    auto factor = sin_angle / angle;

    return Quaternion(
        cos_angle, rotation_vector[0] * factor, rotation_vector[1] * factor,
        rotation_vector[2] * factor
    );
}

std::array<double, 3> rotation_vector_from_quaternion(const Quaternion& quaternion) {
    auto components = quaternion.GetComponents();
    auto sin_angle_squared = components[1] * components[1] + components[2] * components[2] +
                             components[3] * components[3];

    // Return the rotation vector {0, 0, 0} if provided quaternion is null
    if (close_to(sin_angle_squared, 0.)) {
        return {0., 0., 0.};
    }

    double sin_angle = std::sqrt(sin_angle_squared);
    double k = 2. * std::atan2(sin_angle, components[0]) / sin_angle;

    return {components[1] * k, components[2] * k, components[3] * k};
}

}  // namespace openturbine::rigid_pendulum
