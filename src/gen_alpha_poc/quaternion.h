#pragma once

#include <cmath>
#include <tuple>

#include "src/gen_alpha_poc/rotation_matrix.h"
#include "src/gen_alpha_poc/utilities.h"
#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gen_alpha_solver {

/// @brief Class to represent a quaternion
class Quaternion {
public:
    /// Constructs a quaternion based on provided values - if none provided, the quaternion is
    /// initialized to a null quaternion
    KOKKOS_FUNCTION
    Quaternion(double q0 = 0., double q1 = 0., double q2 = 0., double q3 = 0.)
        : q0_(q0), q1_(q1), q2_(q2), q3_(q3) {}

    /// Returns the values of the quaternion
    std::tuple<double, double, double, double> GetComponents() const { return {q0_, q1_, q2_, q3_}; }

    /// Returns the first component of the quaternion
    KOKKOS_FUNCTION
    inline double GetScalarComponent() const { return q0_; }

    /// Returns the second component of the quaternion
    KOKKOS_FUNCTION
    inline double GetXComponent() const { return q1_; }

    /// Returns the third component of the quaternion
    KOKKOS_FUNCTION
    inline double GetYComponent() const { return q2_; }

    /// Returns the fourth component of the quaternion
    KOKKOS_FUNCTION
    inline double GetZComponent() const { return q3_; }

    /// Adds provided quaternion to this quaternion and returns the result
    KOKKOS_FUNCTION
    inline Quaternion operator+(const Quaternion& other) const {
        return Quaternion(
            this->q0_ + other.q0_, this->q1_ + other.q1_, this->q2_ + other.q2_,
            this->q3_ + other.q3_
        );
    }

    /// Subtracts provided quaternion from this quaternion and returns the result
    KOKKOS_FUNCTION
    inline Quaternion operator-(const Quaternion& other) const {
        return Quaternion(
            this->q0_ - other.q0_, this->q1_ - other.q1_, this->q2_ - other.q2_,
            this->q3_ - other.q3_
        );
    }

    /// Multiplies provided quaternion with this quaternion and returns the result
    KOKKOS_FUNCTION
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
    KOKKOS_FUNCTION
    inline Quaternion operator*(double scalar) const {
        return Quaternion(
            this->q0_ * scalar, this->q1_ * scalar, this->q2_ * scalar, this->q3_ * scalar
        );
    }

    /// Divides this quaternion with a scalar and returns the result
    KOKKOS_FUNCTION
    inline Quaternion operator/(double scalar) const {
        return Quaternion(
            this->q0_ / scalar, this->q1_ / scalar, this->q2_ / scalar, this->q3_ / scalar
        );
    }

    /// Returns the length/Euclidean/L2 norm of the quaternion
    KOKKOS_FUNCTION
    inline double Length() const {
        return std::sqrt(
            this->q0_ * this->q0_ + this->q1_ * this->q1_ + this->q2_ * this->q2_ +
            this->q3_ * this->q3_
        );
    }

    /// Returns if the quaternion is a unit quaternion
    KOKKOS_FUNCTION
    bool IsUnitQuaternion() const { return close_to(Length(), 1.); }

    /// Returns a unit quaternion based on the this quaternion
    KOKKOS_FUNCTION
    Quaternion GetUnitQuaternion() const {
        double length = Length();

        if (close_to(length, 1.)) {
            return *this;
        }

        return *this / length;
    }

    /// Returns the conjugate of this quaternion
    KOKKOS_FUNCTION
    inline Quaternion GetConjugate() const {
        return Quaternion(this->q0_, -this->q1_, -this->q2_, -this->q3_);
    }

    /// Returns the inverse of this quaternion
    KOKKOS_FUNCTION
    inline Quaternion GetInverse() const { return GetConjugate() / (Length() * Length()); }

private:
    double q0_;
    double q1_;
    double q2_;
    double q3_;
};

/// Returns a 4-D quaternion from provided 3-D rotation vector, i.e. exponential map
KOKKOS_INLINE_FUNCTION
Quaternion quaternion_from_rotation_vector(const Vector& vector) {
    auto v0 = vector.GetXComponent();
    auto v1 = vector.GetYComponent();
    auto v2 = vector.GetZComponent();

    double angle = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);

    // Return the quaternion {1, 0, 0, 0} if provided rotation vector is null
    if (close_to(angle, 0.)) {
        return Quaternion(1.0, 0.0, 0.0, 0.0);
    }

    double sin_angle = std::sin(angle / 2.0);
    double cos_angle = std::cos(angle / 2.0);
    auto factor = sin_angle / angle;

    return Quaternion(cos_angle, v0 * factor, v1 * factor, v2 * factor);
}

/// Returns a 3-D rotation vector from provided 4-D quaternion, i.e. logarithmic map
KOKKOS_INLINE_FUNCTION
Vector rotation_vector_from_quaternion(const Quaternion& quaternion) {
    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();

    auto sin_angle_squared = q1 * q1 + q2 * q2 + q3 * q3;

    // Return the rotation vector {0, 0, 0} if provided quaternion is null
    if (close_to(sin_angle_squared, 0.)) {
        return Vector{0.0, 0.0, 0.0};
    }

    double sin_angle = std::sqrt(sin_angle_squared);
    double k = 2. * std::atan2(sin_angle, q0) / sin_angle;

    return {q1 * k, q2 * k, q3 * k};
}

/// Returns a quaternion from provided Euler parameters/angle-axis representation of rotation
KOKKOS_INLINE_FUNCTION
Quaternion quaternion_from_angle_axis(double angle, const Vector& axis) {
    auto v0 = axis.GetXComponent();
    auto v1 = axis.GetYComponent();
    auto v2 = axis.GetZComponent();

    double sin_angle = std::sin(angle / 2.0);
    double cos_angle = std::cos(angle / 2.0);

    // We should always get a unit quaternion from the following components
    return Quaternion(cos_angle, v0 * sin_angle, v1 * sin_angle, v2 * sin_angle);
}

/// Returns Euler parameters/angle-axis representation of rotation from provided quaternion
inline std::tuple<double, Vector> angle_axis_from_quaternion(const Quaternion& quaternion) {
    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();

    double angle = 2. * std::atan2(std::sqrt(q1 * q1 + q2 * q2 + q3 * q3), q0);

    // If angle is null, return the angle 0 and axis {1, 0, 0}
    if (close_to(angle, 0.)) {
        return {0., {1., 0., 0.}};
    }

    angle = wrap_angle_to_pi(angle);
    double k = 1. / std::sqrt(q1 * q1 + q2 * q2 + q3 * q3);
    auto normalized_axis = Vector{q1 * k, q2 * k, q3 * k}.GetUnitVector();

    return {angle, normalized_axis};
}

/// Rotates provided vector by provided *unit* quaternion and returns the result
KOKKOS_INLINE_FUNCTION
Vector rotate_vector(const Quaternion& quaternion, const Vector& vector) {
    auto v0 = vector.GetXComponent();
    auto v1 = vector.GetYComponent();
    auto v2 = vector.GetZComponent();

    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();

    return Vector{
        (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * v0 + 2. * (q1 * q2 - q0 * q3) * v1 +
            2. * (q1 * q3 + q0 * q2) * v2,
        2. * (q1 * q2 + q0 * q3) * v0 + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * v1 +
            2. * (q2 * q3 - q0 * q1) * v2,
        2. * (q1 * q3 - q0 * q2) * v0 + 2. * (q2 * q3 + q0 * q1) * v1 +
            (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * v2};
}

/// Converts a 4x1 quaternion to a 3x3 rotation matrix and returns the result
KOKKOS_INLINE_FUNCTION
RotationMatrix quaternion_to_rotation_matrix(const Quaternion& quaternion) {
    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();

    return RotationMatrix{
        q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
        2. * (q1 * q2 - q0 * q3),
        2. * (q1 * q3 + q0 * q2),
        2. * (q1 * q2 + q0 * q3),
        q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
        2. * (q2 * q3 - q0 * q1),
        2. * (q1 * q3 - q0 * q2),
        2. * (q2 * q3 + q0 * q1),
        q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3};
}

/// Converts a 4x1 unit quaternion to a 3x3 rotation matrix and returns the result
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void EulerParameterToRotationMatrix(const Kokkos::View<double[4], MemorySpace> euler_param, Kokkos::View<double[3][3], MemorySpace> rotation_matrix) {
    auto c0 = euler_param(0);
    auto c = Kokkos::subview(euler_param, Kokkos::make_pair(1, 4));

    auto tilde_c = gen_alpha_solver::create_cross_product_matrix(c);
    auto tilde_c_tilde_c = gen_alpha_solver::multiply_matrix_with_matrix(tilde_c, tilde_c);

    for(std::size_t i = 0; i < 3; ++i) {
        for(std::size_t j = 0; j < 3; ++j) {
            rotation_matrix(i, j) = (i == j) + 2 * c0 * tilde_c(i, j) + 2 * tilde_c_tilde_c(i, j);
        }
    }    
}

/// Converts a 3x3 rotation matrix to a 4x1 quaternion and returns the result
KOKKOS_INLINE_FUNCTION
Quaternion rotation_matrix_to_quaternion(const RotationMatrix& rotation_matrix) {
    auto rot_00 = rotation_matrix(0, 0);
    auto rot_11 = rotation_matrix(1, 1);
    auto rot_22 = rotation_matrix(2, 2);
    auto trace = rot_00 + rot_11 + rot_22;

    if (trace > 0) {
        auto s = 0.5 / std::sqrt(trace + 1.0);
        return Quaternion{
            0.25 / s, (rotation_matrix(2, 1) - rotation_matrix(1, 2)) * s,
            (rotation_matrix(0, 2) - rotation_matrix(2, 0)) * s,
            (rotation_matrix(1, 0) - rotation_matrix(0, 1)) * s};
    } else if (rot_00 > rot_11 && rot_00 > rot_22) {
        auto s = 2.0 * std::sqrt(1.0 + rot_00 - rot_11 - rot_22);
        return Quaternion(
            (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / s, 0.25 * s,
            (rotation_matrix(0, 1) + rotation_matrix(1, 0)) / s,
            (rotation_matrix(0, 2) + rotation_matrix(2, 0)) / s
        );
    } else if (rot_11 > rot_22) {
        auto s = 2.0 * std::sqrt(1.0 + rot_11 - rot_00 - rot_22);
        return Quaternion(
            (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / s,
            (rotation_matrix(0, 1) + rotation_matrix(1, 0)) / s, 0.25 * s,
            (rotation_matrix(1, 2) + rotation_matrix(2, 1)) / s
        );
    } else {
        auto s = 2.0 * std::sqrt(1.0 + rot_22 - rot_00 - rot_11);
        return Quaternion(
            (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / s,
            (rotation_matrix(0, 2) + rotation_matrix(2, 0)) / s,
            (rotation_matrix(1, 2) + rotation_matrix(2, 1)) / s, 0.25 * s
        );
    }
}

/// Returns the B derivative matrix given for Euler parameters, i.e. unit quaternions
template<typename MemorySpace>
KOKKOS_INLINE_FUNCTION
void BMatrixForQuaternions(const Quaternion& quaternion, Kokkos::View<double[3][4], MemorySpace> bmatrix) {
    auto q0 = quaternion.GetScalarComponent();
    auto q1 = quaternion.GetXComponent();
    auto q2 = quaternion.GetYComponent();
    auto q3 = quaternion.GetZComponent();
        bmatrix(0, 0) = -q1;
        bmatrix(0, 1) = q0;
        bmatrix(0, 2) = -q3;
        bmatrix(0, 3) = q2;

        bmatrix(1, 0) = -q2;
        bmatrix(1, 1) = q3;
        bmatrix(1, 2) = q0;
        bmatrix(1, 3) = -q1;

        bmatrix(2, 0) = -q3;
        bmatrix(2, 1) = -q2;
        bmatrix(2, 2) = q1;
        bmatrix(2, 3) = q0;
}

}  // namespace openturbine::gen_alpha_solver
