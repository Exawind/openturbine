#pragma once

#include <cmath>
#include <tuple>

namespace openturbine::rigid_pendulum {
/// @brief Class to represent a 3-D vector
class Vector {
public:
    /// Constructs a vector based on provided values - if none provided, the vector is
    /// initialized to a null vector
    Vector(double x = 0., double y = 0., double z = 0.);

    /// Returns the values of the vector
    std::tuple<double, double, double> GetComponents() const { return {x_, y_, z_}; }

    /// Returns the first component of the vector
    inline double GetXComponent() const { return x_; }

    /// Returns the second component of the vector
    inline double GetYComponent() const { return y_; }

    /// Returns the third component of the vector
    inline double GetZComponent() const { return z_; }

    /// Adds provided vector to this vector and returns the result
    inline Vector operator+(const Vector& other) const {
        return Vector(this->x_ + other.x_, this->y_ + other.y_, this->z_ + other.z_);
    }

    /// Subtracts provided vector from this vector and returns the result
    inline Vector operator-(const Vector& other) const {
        return Vector(this->x_ - other.x_, this->y_ - other.y_, this->z_ - other.z_);
    }

    /// Returns dot product of provided vector with this vector
    inline double DotProduct(const Vector& other) const {
        return this->x_ * other.x_ + this->y_ * other.y_ + this->z_ * other.z_;
    }

    /// Returns cross product of provided vector with this vector
    inline Vector CrossProduct(const Vector& other) const {
        return Vector(
            this->y_ * other.z_ - this->z_ * other.y_, this->z_ * other.x_ - this->x_ * other.z_,
            this->x_ * other.y_ - this->y_ * other.x_
        );
    }

private:
    double x_;  ///< First component of the vector
    double y_;  ///< Second component of the vector
    double z_;  ///< Third component of the vector
};

}  // namespace openturbine::rigid_pendulum
