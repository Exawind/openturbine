#pragma once

#include <cmath>
#include <tuple>

namespace openturbine::rigid_pendulum {

// TECH DEBT: We need to introduce a Vector class to represent 3D vectors
using Vector = std::tuple<double, double, double>;

// TODO: Move the following definitions to a constants.h file in a common math directory
static constexpr double kTOLERANCE = 1e-6;
static constexpr double kPI = 3.14159265358979323846;

// TODO: Move the following math related functions to a common math directory
/*!
 * @brief  Returns a boolean indicating if two provided doubles are close to each other
 * @param  a: First double
 * @param  b: Second double
 * @param  epsilon: Tolerance for closeness
 */
bool close_to(double a, double b, double epsilon = kTOLERANCE);

/*!
 * @brief  Takes an angle and returns the equivalent angle in the range [-pi, pi]
 * @param  angle: Angle to be wrapped, in radians
 */
double wrap_angle_to_pi(double angle);

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

/*!
 * @brief Returns a quaternion from provided Euler parameters/axis-angle representation of rotation
 * @param angle Angle of rotation in radians, in radians
 * @param axis Axis of rotation, a unit vector
 * @return Quaternion representing the rotation
 */
Quaternion quaternion_from_axis_angle(double angle, const Vector&);

/*!
 * @brief Returns Euler parameters/axis-angle representation of rotation from provided quaternion
 * @param quaternion Quaternion to be converted
 * @return Tuple of angle of rotation in radians and axis of rotation as a unit vector
 */
std::tuple<double, Vector> axis_angle_from_quaternion(const Quaternion&);

}  // namespace openturbine::rigid_pendulum
