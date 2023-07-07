#include "src/rigid_pendulum_poc/quaternion.h"

#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

bool close_to(double a, double b) {
    a = std::abs(a);
    b = std::abs(b);

    if (a < 1e-6) {
        if (b < 1e-6) {
            return true;
        }
        return false;
    }

    if ((std::abs(a - b) / a) < 1e-6) {
        return true;
    }
    return false;
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

    // Return the quaternion itself if it is already a unit quaternion or null
    if (close_to(length, 0.) || close_to(length, 1.)) {
        return *this;
    }

    return Quaternion(
        values_[0] / length, values_[1] / length, values_[2] / length, values_[3] / length
    );
}

}  // namespace openturbine::rigid_pendulum
