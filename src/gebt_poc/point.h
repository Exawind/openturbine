#pragma once

#include "src/rigid_pendulum_poc/vector.h"

using Vector = openturbine::rigid_pendulum::Vector;

namespace openturbine::gebt_poc {

/// @brief Class to represent a point in 3-D cartesian coordinates
class Point {
public:
    /// Constructs a point based on provided values - if none provided, the point is
    /// initialized to the origin
    KOKKOS_FUNCTION
    Point(double x = 0., double y = 0., double z = 0.) : x_{x}, y_{y}, z_{z} {}

    /// Returns the x component of the point
    KOKKOS_FUNCTION
    inline double GetXComponent() const { return x_; }

    /// Returns the y component of the point
    KOKKOS_FUNCTION
    inline double GetYComponent() const { return y_; }

    /// Returns the z component of the point
    KOKKOS_FUNCTION
    inline double GetZComponent() const { return z_; }

    /// Returns the distance between this point and provided point
    KOKKOS_FUNCTION
    inline double DistanceTo(const Point& other) const {
        return std::sqrt(
            std::pow(this->x_ - other.x_, 2) + std::pow(this->y_ - other.y_, 2) +
            std::pow(this->z_ - other.z_, 2)
        );
    }

    /// Returns the position vector of this point
    KOKKOS_FUNCTION
    inline Vector GetPositionVector() const { return Vector(x_, y_, z_); }

private:
    double x_;  //< x-coordinate
    double y_;  //< y-coordinate
    double z_;  //< z-coordinate
};

}  // namespace openturbine::gebt_poc
