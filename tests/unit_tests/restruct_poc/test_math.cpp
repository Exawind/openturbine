#include <gtest/gtest.h>

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::restruct_poc::tests {

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutXAxis) {
    // 90 degree rotation about the x axis
    auto q = Kokkos::View<double[4]>("q");
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 0.707107;
    q_host(1) = 0.707107;
    q_host(2) = 0.;
    q_host(3) = 0.;
    Kokkos::deep_copy(q, q_host);

    // Convert quaternion to rotation matrix and compare to expected rotation matrix
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    QuaternionToRotationMatrix(q, R_from_q);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        R_from_q, {{1., 0., 0.}, {0., 0., -1.}, {0., 1., 0.}}
    );
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutYAxis) {
    // 90 degree rotation about the y axis
    auto q = Kokkos::View<double[4]>("q");
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 0.707107;
    q_host(1) = 0.;
    q_host(2) = 0.707107;
    q_host(3) = 0.;
    Kokkos::deep_copy(q, q_host);

    // Convert quaternion to rotation matrix and compare to expected rotation matrix
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    QuaternionToRotationMatrix(q, R_from_q);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        R_from_q, {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., 0.}}
    );
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutZAxis) {
    // 90 degree rotation about the z axis
    auto q = Kokkos::View<double[4]>("q");
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 0.707107;
    q_host(1) = 0.;
    q_host(2) = 0.;
    q_host(3) = 0.707107;
    Kokkos::deep_copy(q, q_host);

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
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 0.707107;
    q_host(1) = 0.;
    q_host(2) = 0.707107;
    q_host(3) = 0.;
    Kokkos::deep_copy(q, q_host);

    // Initialize a vector along the x axis
    auto v = Kokkos::View<double[3]>("v");
    auto v_host = Kokkos::create_mirror(v);
    v_host(0) = 1.;
    v_host(1) = 0.;
    v_host(2) = 0.;
    Kokkos::deep_copy(v, v_host);

    // Rotate the x axis 90 degrees about the y axis and compare to expected result
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    RotateVectorByQuaternion(q, v, v_rot);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v_rot, {0., 0., -1.});
}

TEST(QuaternionTest, RotateZAxis90DegreesAboutXAxis) {
    // Quaternion representing a 90 degree rotation about the x axis
    auto q = Kokkos::View<double[4]>("q");
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 0.707107;
    q_host(1) = 0.707107;
    q_host(2) = 0.;
    q_host(3) = 0.;
    Kokkos::deep_copy(q, q_host);

    // Initialize a vector along the z axis
    auto v = Kokkos::View<double[3]>("v");
    auto v_host = Kokkos::create_mirror(v);
    v_host(0) = 0.;
    v_host(1) = 0.;
    v_host(2) = 1.;
    Kokkos::deep_copy(v, v_host);

    // Rotate the z axis 90 degrees about the x axis and compare to expected result
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    RotateVectorByQuaternion(q, v, v_rot);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v_rot, {0., -1., 0.});
}

TEST(QuaternionTest, RotateXAxis45DegreesAboutZAxis) {
    // Quaternion representing a 45 degree rotation about the z axis
    auto q = Kokkos::View<double[4]>("q");
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 0.92388;
    q_host(1) = 0.;
    q_host(2) = 0.;
    q_host(3) = 0.382683;
    Kokkos::deep_copy(q, q_host);

    // Initialize a vector along the x axis
    auto v = Kokkos::View<double[3]>("v");
    auto v_host = Kokkos::create_mirror(v);
    v_host(0) = 1.;
    v_host(1) = 0.;
    v_host(2) = 0.;
    Kokkos::deep_copy(v, v_host);

    // Rotate the x axis 45 degrees about the z axis and compare to expected result
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    RotateVectorByQuaternion(q, v, v_rot);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v_rot, {0.707107, 0.707107, 0.});
}

TEST(QuaternionTest, RotateXAxisNeg45DegreesAboutZAxis) {
    // Quaternion representing a -45 degree rotation about the z axis
    auto q = Kokkos::View<double[4]>("q");
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 0.92388;
    q_host(1) = 0.;
    q_host(2) = 0.;
    q_host(3) = -0.382683;
    Kokkos::deep_copy(q, q_host);

    // Initialize a vector along the x axis
    auto v = Kokkos::View<double[3]>("v");
    auto v_host = Kokkos::create_mirror(v);
    v_host(0) = 1.;
    v_host(1) = 0.;
    v_host(2) = 0.;
    Kokkos::deep_copy(v, v_host);

    // Rotate the x axis -45 degrees about the z axis and compare to expected result
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    RotateVectorByQuaternion(q, v, v_rot);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v_rot, {0.707107, -0.707107, 0.});
}

TEST(QuaternionTest, QuaternionDerivative) {
    auto q = Kokkos::View<double[4]>("q");
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 1.;
    q_host(1) = 2.;
    q_host(2) = 3.;
    q_host(3) = 4.;
    Kokkos::deep_copy(q, q_host);

    auto m = Kokkos::View<double[3][4]>("m");
    QuaternionDerivative(q, m);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        m, {{-2., 1., -4., 3.}, {-3., 4., 1., -2.}, {-4., -3., 2., 1.}}
    );
}

TEST(QuaternionTest, GetInverse) {
    auto q = Kokkos::View<double[4]>("q");
    auto q_host = Kokkos::create_mirror(q);
    q_host(0) = 1. / std::sqrt(30.);
    q_host(1) = 2. / std::sqrt(30.);
    q_host(2) = 3. / std::sqrt(30.);
    q_host(3) = 4. / std::sqrt(30.);
    Kokkos::deep_copy(q, q_host);

    auto q_inv = Kokkos::View<double[4]>("q_inv");
    QuaternionInverse(q, q_inv);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        q_inv,
        {1. / std::sqrt(30.), -2. / std::sqrt(30.), -3. / std::sqrt(30.), -4. / std::sqrt(30.)}
    );
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set1) {
    auto q1 = Kokkos::View<double[4]>("q1");
    auto q1_host = Kokkos::create_mirror(q1);
    q1_host(0) = 3.;
    q1_host(1) = 1.;
    q1_host(2) = -2.;
    q1_host(3) = 1.;
    Kokkos::deep_copy(q1, q1_host);

    auto q2 = Kokkos::View<double[4]>("q2");
    auto q2_host = Kokkos::create_mirror(q2);
    q2_host(0) = 2.;
    q2_host(1) = -1.;
    q2_host(2) = 2.;
    q2_host(3) = 3.;
    Kokkos::deep_copy(q2, q2_host);

    auto qn = Kokkos::View<double[4]>("qn");
    QuaternionCompose(q1, q2, qn);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(qn, {8., -9., -2., 11.});
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set2) {
    auto q1 = Kokkos::View<double[4]>("q1");
    auto q1_host = Kokkos::create_mirror(q1);
    q1_host(0) = 1.;
    q1_host(1) = 2.;
    q1_host(2) = 3.;
    q1_host(3) = 4.;
    Kokkos::deep_copy(q1, q1_host);

    auto q2 = Kokkos::View<double[4]>("q2");
    auto q2_host = Kokkos::create_mirror(q2);
    q2_host(0) = 5.;
    q2_host(1) = 6.;
    q2_host(2) = 7.;
    q2_host(3) = 8.;
    Kokkos::deep_copy(q2, q2_host);

    auto qn = Kokkos::View<double[4]>("qn");
    QuaternionCompose(q1, q2, qn);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(qn, {-60., 12., 30., 24.});
}

TEST(QuaternionTest, AxialVectorOfMatrix) {
    auto m = Kokkos::View<double[3][3]>("m");
    auto m_host = Kokkos::create_mirror(m);
    m_host(0, 0) = 0.;
    m_host(0, 1) = -1.;
    m_host(0, 2) = 0.;
    m_host(1, 0) = 1.;
    m_host(1, 1) = 0.;
    m_host(1, 2) = 0.;
    m_host(2, 0) = 0.;
    m_host(2, 1) = 0.;
    m_host(2, 2) = 0.;
    Kokkos::deep_copy(m, m_host);

    auto v = Kokkos::View<double[3]>("v");
    AxialVectorOfMatrix(m, v);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(v, {0., 0., 1.});
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set1) {
    auto phi = Kokkos::View<double[3]>("phi");
    auto phi_host = Kokkos::create_mirror(phi);
    phi_host(0) = 1.;
    phi_host(1) = 2.;
    phi_host(2) = 3.;
    Kokkos::deep_copy(phi, phi_host);

    auto quaternion = Kokkos::View<double[4]>("quaternion");
    RotationVectorToQuaternion(phi, quaternion);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        quaternion, {-0.295551, 0.255322, 0.510644, 0.765966}
    );
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set2) {
    auto phi = Kokkos::View<double[3]>("phi");
    auto phi_host = Kokkos::create_mirror(phi);
    phi_host(0) = 0.;
    phi_host(1) = 0.;
    phi_host(2) = 1.570796;
    Kokkos::deep_copy(phi, phi_host);

    auto quaternion = Kokkos::View<double[4]>("quaternion");
    RotationVectorToQuaternion(phi, quaternion);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(quaternion, {0.707107, 0., 0., 0.707107});
}

TEST(VectorTest, VecTilde) {
    auto v = Kokkos::View<double[3]>("v");
    auto v_host = Kokkos::create_mirror(v);
    v_host(0) = 1.;
    v_host(1) = 2.;
    v_host(2) = 3.;
    Kokkos::deep_copy(v, v_host);

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
