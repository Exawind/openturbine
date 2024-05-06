#include <gtest/gtest.h>

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::restruct_poc::tests {

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_1) {
    // 90 degree rotation about the x axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.707107;
            q(1) = 0.707107;
            q(2) = 0.;
            q(3) = 0.;
        }
    );

    // Convert quaternion to rotation matrix and compare to expected rotation matrix
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    QuaternionToRotationMatrix(q, R_from_q);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        R_from_q, {{1., 0., 0.}, {0., 0., -1.}, {0., 1., 0.}}
    );
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_2) {
    // 90 degree rotation about the y axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.707107;
            q(1) = 0.;
            q(2) = 0.707107;
            q(3) = 0.;
        }
    );

    // Convert quaternion to rotation matrix and compare to expected rotation matrix
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    QuaternionToRotationMatrix(q, R_from_q);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        R_from_q, {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., 0.}}
    );
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_3) {
    // 90 degree rotation about the z axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.707107;
            q(1) = 0.;
            q(2) = 0.;
            q(3) = 0.707107;
        }
    );

    // Convert quaternion to rotation matrix and compare to expected rotation matrix
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    QuaternionToRotationMatrix(q, R_from_q);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        R_from_q, {{0., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}}
    );
}

TEST(QuaternionTest, RotateXAxis90DegreesAboutYAxis) {
    // Quaternion representing a 90 degree rotation about the y axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.707107;
            q(1) = 0.;
            q(2) = 0.707107;
            q(3) = 0.;
        }
    );

    // Initialize a vector along the x axis
    auto v = Kokkos::View<double[3]>("v");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            v(0) = 1.;
            v(1) = 0.;
            v(2) = 0.;
        }
    );

    // Rotate the x axis 90 degrees about the y axis and compare to expected result
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    RotateVectorByQuaternion(q, v, v_rot);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v_rot, {0., 0., -1.});
}

TEST(QuaternionTest, RotateZAxis90DegreesAboutXAxis) {
    // Quaternion representing a 90 degree rotation about the x axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.707107;
            q(1) = 0.707107;
            q(2) = 0.;
            q(3) = 0.;
        }
    );

    // Initialize a vector along the z axis
    auto v = Kokkos::View<double[3]>("v");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            v(0) = 0.;
            v(1) = 0.;
            v(2) = 1.;
        }
    );

    // Rotate the z axis 90 degrees about the x axis and compare to expected result
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    RotateVectorByQuaternion(q, v, v_rot);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v_rot, {0., -1., 0.});
}

TEST(QuaternionTest, RotateXAxis45DegreesAboutZAxis) {
    // Quaternion representing a 45 degree rotation about the z axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.92388;
            q(1) = 0.;
            q(2) = 0.;
            q(3) = 0.382683;
        }
    );

    // Initialize a vector along the x axis
    auto v = Kokkos::View<double[3]>("v");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            v(0) = 1.;
            v(1) = 0.;
            v(2) = 0.;
        }
    );

    // Rotate the x axis 45 degrees about the z axis and compare to expected result
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    RotateVectorByQuaternion(q, v, v_rot);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v_rot, {0.707107, 0.707107, 0.});
}

TEST(QuaternionTest, RotateXAxisNeg45DegreesAboutZAxis) {
    // Quaternion representing a -45 degree rotation about the z axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.92388;
            q(1) = 0.;
            q(2) = 0.;
            q(3) = -0.382683;
        }
    );

    // Initialize a vector along the x axis
    auto v = Kokkos::View<double[3]>("v");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            v(0) = 1.;
            v(1) = 0.;
            v(2) = 0.;
        }
    );

    // Rotate the x axis -45 degrees about the z axis and compare to expected result
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    RotateVectorByQuaternion(q, v, v_rot);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v_rot, {0.707107, -0.707107, 0.});
}

TEST(QuaternionTest, GetInverse) {
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 1. / std::sqrt(30.);
            q(1) = 2. / std::sqrt(30.);
            q(2) = 3. / std::sqrt(30.);
            q(3) = 4. / std::sqrt(30.);
        }
    );

    auto q_inv = Kokkos::View<double[4]>("q_inv");
    QuaternionInverse(q, q_inv);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        q_inv,
        {1. / std::sqrt(30.), -2. / std::sqrt(30.), -3. / std::sqrt(30.), -4. / std::sqrt(30.)}
    );
}

}  // namespace openturbine::restruct_poc::tests
