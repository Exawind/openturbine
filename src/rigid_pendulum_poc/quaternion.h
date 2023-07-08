#pragma once

#include <array>

namespace openturbine::rigid_pendulum {

/// Returns a boolean indicating if two provided doubles are close to each other
bool close_to(double a, double b);

/// @brief Class to represent a quaternion
class Quaternion {
public:
    /// Constructs a quaternion based on an array of values - if none are provided, the quaternion is
    /// initialized to a null quaternion
    Quaternion(std::array<double, 4> values = {0.0, 0.0, 0.0, 0.0});

    /// Constructs a quaternion based on four scalar values
    Quaternion(double, double, double, double);

    /// Construct a quaternion based on a scalar value and a vector
    Quaternion(double, const std::array<double, 3>&);

    /// Returns the values of the quaternion
    std::array<double, 4> GetComponents() const { return values_; }

    /// Returns the value of the quaternion at a given index
    inline double operator[](size_t index) const {
        if (index <= 3) {
            return values_[index];
        } else {
            throw std::out_of_range("Quaternion index out of range");
        }
    }

    /// Returns the first value of the quaternion
    inline double GetScalarComponent() const { return values_[0]; }

    /// Returns the last three values of the quaternion as a vector
    inline std::array<double, 3> GetVectorComponent() const {
        return {values_[1], values_[2], values_[3]};
    }

    /// Adds two quaternions and returns the result
    inline Quaternion operator+(const Quaternion& other) const {
        return Quaternion(
            this->values_[0] + other.values_[0], this->values_[1] + other.values_[1],
            this->values_[2] + other.values_[2], this->values_[3] + other.values_[3]
        );
    }

    /// Subtracts two quaternions and returns the result
    inline Quaternion operator-(const Quaternion& other) const {
        return Quaternion(
            this->values_[0] - other.values_[0], this->values_[1] - other.values_[1],
            this->values_[2] - other.values_[2], this->values_[3] - other.values_[3]
        );
    }

    /// Multiplies two quaternions and returns the result
    inline Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            this->values_[0] * other.values_[0] - this->values_[1] * other.values_[1] -
                this->values_[2] * other.values_[2] - this->values_[3] * other.values_[3],
            this->values_[0] * other.values_[1] + this->values_[1] * other.values_[0] +
                this->values_[2] * other.values_[3] - this->values_[3] * other.values_[2],
            this->values_[0] * other.values_[2] - this->values_[1] * other.values_[3] +
                this->values_[2] * other.values_[0] + this->values_[3] * other.values_[1],
            this->values_[0] * other.values_[3] + this->values_[1] * other.values_[2] -
                this->values_[2] * other.values_[1] + this->values_[3] * other.values_[0]
        );
    }

    /// Multiplies the quaternion by a scalar and returns the result
    inline Quaternion operator*(double scalar) const {
        return Quaternion(
            values_[0] * scalar, values_[1] * scalar, values_[2] * scalar, values_[3] * scalar
        );
    }

    /// Divides the quaternion by a scalar and returns the result
    inline Quaternion operator/(double scalar) const {
        return Quaternion(
            values_[0] / scalar, values_[1] / scalar, values_[2] / scalar, values_[3] / scalar
        );
    }

    /// Returns the length/Euclidean/L2 norm of the quaternion
    inline double Length() const {
        return std::sqrt(
            values_[0] * values_[0] + values_[1] * values_[1] + values_[2] * values_[2] +
            values_[3] * values_[3]
        );
    }

    /// Returns if the quaternion is a unit quaternion
    bool IsUnitQuaternion() const;

    /// Returns a unit quaternion based on the current quaternion
    Quaternion GetUnitQuaternion() const;

    /// Returns the conjugate of the quaternion
    inline Quaternion GetConjugate() const {
        return Quaternion(values_[0], -values_[1], -values_[2], -values_[3]);
    }

    /// Returns the inverse of the quaternion
    inline Quaternion GetInverse() const { return GetConjugate() / (Length() * Length()); }

private:
    std::array<double, 4> values_;
};

/// Returns a 4-D quaternion from provided 3-D rotation vector
Quaternion quaternion_from_rotation_vector(const std::array<double, 3>& rotation_vector);

/// Returns a 3-D rotation vector from provided 4-D quaternion
std::array<double, 3> rotation_vector_from_quaternion(const Quaternion& quaternion);

}  // namespace openturbine::rigid_pendulum
