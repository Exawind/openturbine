#include "src/rigid_pendulum_poc/quaternion.h"

#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

Quaternion::Quaternion(std::array<double, 4> values) : values_(std::move(values)) {
}

Quaternion::Quaternion(double p0, double p1, double p2, double p3) : values_({p0, p1, p2, p3}) {
}

Quaternion::Quaternion(double scalar, const std::array<double, 3>& vector)
    : values_({scalar, vector[0], vector[1], vector[2]}) {
}

}  // namespace openturbine::rigid_pendulum
