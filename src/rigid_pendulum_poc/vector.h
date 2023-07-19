#pragma once

#include <cmath>
#include <stdexcept>
#include <tuple>

#include "src/rigid_pendulum_poc/utilities.h"

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

    /// Multiplies provided scalar with this vector and returns the result, i.e. element-wise
    /// multiplication
    inline Vector operator*(double scalar) const {
        return Vector(this->x_ * scalar, this->y_ * scalar, this->z_ * scalar);
    }

    /// Divides this vector by provided scalar and returns the result, i.e. element-wise
    /// division
    inline Vector operator/(double scalar) const {
        return Vector(this->x_ / scalar, this->y_ / scalar, this->z_ / scalar);
    }

    /// Returns if this vector is close to provided vector, i.e. if the difference between
    /// their components is less than a small number
    inline bool operator==(const Vector& other) const {
        return close_to(this->x_, other.x_) && close_to(this->y_, other.y_) &&
               close_to(this->z_, other.z_);
    }

    /// Returns the magnitude/length/Euclidean norm of the vector
    inline double Length() const {
        return std::sqrt(this->x_ * this->x_ + this->y_ * this->y_ + this->z_ * this->z_);
    }

    /// Returns if the vector is a unit vector, i.e. its length is 1
    inline bool IsUnitVector() const { return close_to(this->Length(), 1.); }

    /// Returns if the vector is a null vector, i.e. its length is 0
    inline bool IsNullVector() const { return close_to(this->Length(), 0.); }

    /// Returns a unit vector in the same direction as this vector
    inline Vector GetUnitVector() const {
        if (this->IsNullVector()) {
            throw std::runtime_error("Cannot get unit vector of null vector");
        }
        return *this / this->Length();
    }

    /// Calculates the dot product of provided vector with this vector
    inline double DotProduct(const Vector& other) const {
        return this->x_ * other.x_ + this->y_ * other.y_ + this->z_ * other.z_;
    }

    /// Calculates the cross product of provided vector with this vector
    inline Vector CrossProduct(const Vector& other) const {
        return Vector(
            this->y_ * other.z_ - this->z_ * other.y_, this->z_ * other.x_ - this->x_ * other.z_,
            this->x_ * other.y_ - this->y_ * other.x_
        );
    }

    /// Returns if this vector is normal to provided vector, i.e. if their dot product is 0
    inline bool IsNormalTo(const Vector& other) const {
        return close_to(this->DotProduct(other), 0.);
    }

    /// Returns if this vector is parallel to provided vector, i.e. if their cross product is 0
    inline bool IsParallelTo(const Vector& other) const {
        return this->CrossProduct(other).IsNullVector();
    }

private:
    double x_;  ///< First component of the vector
    double y_;  ///< Second component of the vector
    double z_;  ///< Third component of the vector
};

}  // namespace openturbine::rigid_pendulum
