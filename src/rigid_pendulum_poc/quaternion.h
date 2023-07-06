#pragma once

#include <array>

namespace openturbine::rigid_pendulum {

/// @brief Class to represent a quaternion
class Quaternion {
public:
    /// @brief Constructs a quaternion based on an array of values, if none are
    ///        provided, the quaternion is initialized to the null quaternion
    Quaternion(std::array<double, 4> values = {0.0, 0.0, 0.0, 0.0});

    /// @brief Constructs a quaternion based on four scalar values
    Quaternion(double, double, double, double);

    /// @brief  Construct a quaternion based on a scalar value and a vector
    Quaternion(double, const std::array<double, 3>&);

    /// Returns the values of the quaternion
    std::array<double, 4> values() const { return values_; }

    /// Returns the value of the quaternion at a given index
    inline double operator[](size_t index) const {
        if (index <= 3) {
            return values_[index];
        } else {
            throw std::out_of_range("Quaternion index out of range");
        }
    }

    /// Returns the length/Euclidean norm of the quaternion
    inline double length() const {
        return std::sqrt(
            values_[0] * values_[0] + values_[1] * values_[1] + values_[2] * values_[2] +
            values_[3] * values_[3]
        );
    }

private:
    std::array<double, 4> values_;
};

}  // namespace openturbine::rigid_pendulum
