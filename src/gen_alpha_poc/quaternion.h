#pragma once

#include <cmath>
#include <tuple>

#include "src/gen_alpha_poc/utilities.h"
#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gen_alpha_solver {

// TODO: Refactor to create a Matrix class
using RotationMatrix = std::tuple<Vector, Vector, Vector>;

/// @brief Class to represent a quaternion
class Quaternion {
public:
    /// Constructs a quaternion based on provided values - if none provided, the quaternion is
    /// initialized to a null quaternion
    Quaternion(double q0 = 0., double q1 = 0., double q2 = 0., double q3 = 0.);

    /// Returns the values of the quaternion
    std::tuple<double, double, double, double> GetComponents() const { return {q0_, q1_, q2_, q3_}; }

    /// Returns the first component of the quaternion
    inline double GetScalarComponent() const { return q0_; }

    /// Returns the second component of the quaternion
    inline double GetXComponent() const { return q1_; }

    /// Returns the third component of the quaternion
    inline double GetYComponent() const { return q2_; }

    /// Returns the fourth component of the quaternion
    inline double GetZComponent() const { return q3_; }

    /// Adds provided quaternion to this quaternion and returns the result
    inline Quaternion operator+(const Quaternion& other) const {
        return Quaternion(
            this->q0_ + other.q0_, this->q1_ + other.q1_, this->q2_ + other.q2_,
            this->q3_ + other.q3_
        );
    }

    /// Subtracts provided quaternion from this quaternion and returns the result
    inline Quaternion operator-(const Quaternion& other) const {
        return Quaternion(
            this->q0_ - other.q0_, this->q1_ - other.q1_, this->q2_ - other.q2_,
            this->q3_ - other.q3_
        );
    }

    /// Multiplies provided quaternion with this quaternion and returns the result
    inline Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            this->q0_ * other.q0_ - this->q1_ * other.q1_ - this->q2_ * other.q2_ -
                this->q3_ * other.q3_,
            this->q0_ * other.q1_ + this->q1_ * other.q0_ + this->q2_ * other.q3_ -
                this->q3_ * other.q2_,
            this->q0_ * other.q2_ - this->q1_ * other.q3_ + this->q2_ * other.q0_ +
                this->q3_ * other.q1_,
            this->q0_ * other.q3_ + this->q1_ * other.q2_ - this->q2_ * other.q1_ +
                this->q3_ * other.q0_
        );
    }

    /// Multiplies this quaternion with a scalar and returns the result
    inline Quaternion operator*(double scalar) const {
        return Quaternion(
            this->q0_ * scalar, this->q1_ * scalar, this->q2_ * scalar, this->q3_ * scalar
        );
    }

    /// Divides this quaternion with a scalar and returns the result
    inline Quaternion operator/(double scalar) const {
        return Quaternion(
            this->q0_ / scalar, this->q1_ / scalar, this->q2_ / scalar, this->q3_ / scalar
        );
    }

    /// Returns the length/Euclidean/L2 norm of the quaternion
    inline double Length() const {
        return std::sqrt(
            this->q0_ * this->q0_ + this->q1_ * this->q1_ + this->q2_ * this->q2_ +
            this->q3_ * this->q3_
        );
    }

    /// Returns if the quaternion is a unit quaternion
    bool IsUnitQuaternion() const;

    /// Returns a unit quaternion based on the this quaternion
    Quaternion GetUnitQuaternion() const;

    /// Returns the conjugate of this quaternion
    inline Quaternion GetConjugate() const {
        return Quaternion(this->q0_, -this->q1_, -this->q2_, -this->q3_);
    }

    /// Returns the inverse of this quaternion
    inline Quaternion GetInverse() const { return GetConjugate() / (Length() * Length()); }

private:
    double q0_;
    double q1_;
    double q2_;
    double q3_;
};

/// Returns a 4-D quaternion from provided 3-D rotation vector, i.e. exponential map
Quaternion quaternion_from_rotation_vector(const Vector&);

/// Returns a 3-D rotation vector from provided 4-D quaternion, i.e. logarithmic map
Vector rotation_vector_from_quaternion(const Quaternion&);

/// Returns a quaternion from provided Euler parameters/angle-axis representation of rotation
Quaternion quaternion_from_angle_axis(double, const Vector&);

/// Returns Euler parameters/angle-axis representation of rotation from provided quaternion
std::tuple<double, Vector> angle_axis_from_quaternion(const Quaternion&);

/// Rotates provided 3 x 1 vector by provided 4 x 1 *unit* quaternion and returns the result
Vector rotate_vector(const Quaternion&, const Vector&);

/// Converts a 4 x 1 quaternion to a 3 x 3 rotation matrix and returns the result
RotationMatrix quaternion_to_rotation_matrix(const Quaternion&);

/// Converts a 3 x 3 rotation matrix to a 4 x 1 quaternion and returns the result
Quaternion rotation_matrix_to_quaternion(const RotationMatrix&);

}  // namespace openturbine::gen_alpha_solver
