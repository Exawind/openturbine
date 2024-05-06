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

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set1) {
    auto q1 = Kokkos::View<double[4]>("q1");
    auto q2 = Kokkos::View<double[4]>("q2");
    auto qn = Kokkos::View<double[4]>("qn");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q1(0) = 3.;
            q1(1) = 1.;
            q1(2) = -2.;
            q1(3) = 1.;
            q2(0) = 2.;
            q2(1) = -1.;
            q2(2) = 2.;
            q2(3) = 3.;
        }
    );

    QuaternionCompose(q1, q2, qn);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(qn, {8., -9., -2., 11.});
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set2) {
    auto q1 = Kokkos::View<double[4]>("q1");
    auto q2 = Kokkos::View<double[4]>("q2");
    auto qn = Kokkos::View<double[4]>("qn");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q1(0) = 1.;
            q1(1) = 2.;
            q1(2) = 3.;
            q1(3) = 4.;
            q2(0) = 5.;
            q2(1) = 6.;
            q2(2) = 7.;
            q2(3) = 8.;
        }
    );

    QuaternionCompose(q1, q2, qn);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(qn, {-60., 12., 30., 24.});
}

TEST(QuaternionTest, ComputeAxialVector) {
    auto m = Kokkos::View<double[3][3]>("m");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            m(0, 0) = 0.;
            m(0, 1) = -1.;
            m(0, 2) = 0.;
            m(1, 0) = 1.;
            m(1, 1) = 0.;
            m(1, 2) = 0.;
            m(2, 0) = 0.;
            m(2, 1) = 0.;
            m(2, 2) = 0.;
        }
    );

    auto v = Kokkos::View<double[3]>("v");
    ComputeAxialVector(m, v);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v, {0., 0., 1.});
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set1) {
    auto phi = Kokkos::View<double[3]>("phi");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            phi(0) = 1.;
            phi(1) = 2.;
            phi(2) = 3.;
        }
    );

    auto quaternion = Kokkos::View<double[4]>("quaternion");
    RotationVectorToQuaternion(phi, quaternion);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        quaternion, {-0.295551, 0.255322, 0.510644, 0.765966}
    );
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set2) {
    auto phi = Kokkos::View<double[3]>("phi");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            phi(0) = 0.;
            phi(1) = 0.;
            phi(2) = 1.570796;
        }
    );

    auto quaternion = Kokkos::View<double[4]>("quaternion");
    RotationVectorToQuaternion(phi, quaternion);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(quaternion, {0.707107, 0., 0., 0.707107});
}

TEST(VectorTest, VecTilde) {
    auto v = Kokkos::View<double[3]>("v");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            v(0) = 1.;
            v(1) = 2.;
            v(2) = 3.;
        }
    );

    auto m = Kokkos::View<double[3][3]>("m");
    VecTilde(v, m);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        m, {{0., -3., 2.}, {3., 0., -1.}, {-2., 1., 0.}}
    );
}

TEST(QuaternionTest, CrossProduct_Set1) {
    auto a = std::array<double, 3>{1., 2., 3.};
    auto b = std::array<double, 3>{4., 5., 6.};
    auto c = CrossProduct(a, b);

    ASSERT_EQ(c[0], -3.);
    ASSERT_EQ(c[1], 6.);
    ASSERT_EQ(c[2], -3.);
}

TEST(VectorTest, CrossProduct_Set2) {
    auto a = std::array<double, 3>{0.19, -5.03, 2.71};
    auto b = std::array<double, 3>{1.16, 0.09, 0.37};
    auto c = CrossProduct(a, b);

    ASSERT_EQ(c[0], -5.03 * 0.37 - 2.71 * 0.09);
    ASSERT_EQ(c[1], 2.71 * 1.16 - 0.19 * 0.37);
    ASSERT_EQ(c[2], 0.19 * 0.09 - -5.03 * 1.16);
}

}  // namespace openturbine::restruct_poc::tests
