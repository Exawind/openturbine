#include <numbers>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "math/quaternion_operations.hpp"

namespace openturbine::tests {

template <unsigned size>
auto Create1DView(const std::array<double, size>& input) {
    auto view = Kokkos::View<double[size]>("view");
    auto view_host = Kokkos::View<const double[size], Kokkos::HostSpace>(input.data());
    Kokkos::deep_copy(view, view_host);
    return view;
}

Kokkos::View<double[3][3]> TestQuaternionToRotationMatrix(
    const Kokkos::View<double[4]>::const_type& q
) {
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    Kokkos::parallel_for(
        "QuaternionToRotationMatrix", 1,
        KOKKOS_LAMBDA(int) { math::QuaternionToRotationMatrix(q, R_from_q); }
    );
    return R_from_q;
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutXAxis) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    const auto rotation_x_axis = Create1DView<4>(std::array{inv_sqrt2, inv_sqrt2, 0., 0.});

    const auto R_from_q = TestQuaternionToRotationMatrix(rotation_x_axis);

    const auto R_from_q_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R_from_q);
    constexpr auto expected_data = std::array{1., 0., 0., 0., 0., -1., 0., 1., 0.};
    const auto expected = Kokkos::View<const double[3][3], Kokkos::HostSpace>(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(R_from_q_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutYAxis) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    const auto rotation_x_axis = Create1DView<4>(std::array{inv_sqrt2, 0., inv_sqrt2, 0.});

    const auto R_from_q = TestQuaternionToRotationMatrix(rotation_x_axis);

    const auto R_from_q_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R_from_q);
    constexpr auto expected_data = std::array{0., 0., 1., 0., 1., 0., -1., 0., 0.};
    const auto expected = Kokkos::View<const double[3][3], Kokkos::HostSpace>(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(R_from_q_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutZAxis) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    const auto rotation_x_axis = Create1DView<4>(std::array{inv_sqrt2, 0., 0., inv_sqrt2});

    const auto R_from_q = TestQuaternionToRotationMatrix(rotation_x_axis);

    const auto R_from_q_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R_from_q);
    constexpr auto expected_data = std::array{0., -1., 0., 1., 0., 0., 0., 0., 1.};
    const auto expected = Kokkos::View<const double[3][3], Kokkos::HostSpace>(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(R_from_q_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

TEST(QuaternionTest, ConvertRotationMatrixToQuaternion) {
    const auto n = 25U;
    const auto dtheta = M_PI / static_cast<double>(n);
    for (auto i = 0U; i < n; ++i) {
        for (auto j = 0U; j < n; ++j) {
            auto q_ref = math::RotationVectorToQuaternion(
                {static_cast<double>(i) * dtheta, static_cast<double>(j) * dtheta, 0.}
            );
            auto r = math::QuaternionToRotationMatrix(q_ref);
            auto q_new = math::RotationMatrixToQuaternion(r);
            for (auto m = 0U; m < 4; ++m) {
                EXPECT_NEAR(q_ref[m], q_new[m], 1e-12);
            }
        }
    }
}

Kokkos::View<double[3]> TestRotateVectorByQuaternion(
    const Kokkos::View<double[4]>::const_type& q, const Kokkos::View<double[3]>::const_type& v
) {
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    Kokkos::parallel_for(
        "RotateVectorBoyQuaternion", 1,
        KOKKOS_LAMBDA(int) { math::RotateVectorByQuaternion(q, v, v_rot); }
    );
    return v_rot;
}

TEST(QuaternionTest, RotateYAxisByIdentity) {
    auto rotation_identity = Create1DView<4>({1., 0., 0., 0.});
    auto y_axis = Create1DView<3>({0., 1., 0.});

    const auto v_rot = TestRotateVectorByQuaternion(rotation_identity, y_axis);

    const auto v_rot_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_rot);

    constexpr auto expected_data = std::array{0., 1., 0.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotateXAxis90DegreesAboutYAxis) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    auto rotation_y_axis = Create1DView<4>({inv_sqrt2, 0., inv_sqrt2, 0.});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    const auto v_rot = TestRotateVectorByQuaternion(rotation_y_axis, x_axis);

    const auto v_rot_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_rot);

    constexpr auto expected_data = std::array{0., 0., -1.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotateZAxis90DegreesAboutXAxis) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    auto rotation_x_axis = Create1DView<4>({inv_sqrt2, inv_sqrt2, 0., 0.});
    auto z_axis = Create1DView<3>({0., 0., 1.});
    const auto v_rot = TestRotateVectorByQuaternion(rotation_x_axis, z_axis);

    const auto v_rot_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_rot);

    constexpr auto expected_data = std::array{0., -1., 0.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotateXAxis45DegreesAboutZAxis) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    const auto cos_pi_8 = std::cos(M_PI / 8.);
    const auto sin_pi_8 = std::sin(M_PI / 8.);
    auto rotation_z_axis = Create1DView<4>({cos_pi_8, 0., 0., sin_pi_8});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    const auto v_rot = TestRotateVectorByQuaternion(rotation_z_axis, x_axis);

    const auto v_rot_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_rot);

    const auto expected_data = std::array{inv_sqrt2, inv_sqrt2, 0.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotateXAxisNeg45DegreesAboutZAxis) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    const auto cos_pi_8 = std::cos(M_PI / 8.);
    const auto sin_pi_8 = std::sin(M_PI / 8.);
    auto rotation_z_axis = Create1DView<4>({cos_pi_8, 0., 0., -sin_pi_8});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    const auto v_rot = TestRotateVectorByQuaternion(rotation_z_axis, x_axis);

    const auto v_rot_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_rot);

    const auto expected_data = std::array{inv_sqrt2, -inv_sqrt2, 0.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

Kokkos::View<double[3][4]> TestQuaternionDerivative(const Kokkos::View<double[4]>::const_type& q) {
    auto m = Kokkos::View<double[3][4]>("m");
    Kokkos::parallel_for(
        "QuaternionDerivative", 1, KOKKOS_LAMBDA(int) { math::QuaternionDerivative(q, m); }
    );
    return m;
}

TEST(QuaternionTest, QuaternionDerivative) {
    const auto q = Create1DView<4>({1., 2., 3., 4.});
    const auto derivative = TestQuaternionDerivative(q);

    const auto derivative_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), derivative);

    constexpr auto expected_data = std::array{-2., 1., -4., 3., -3., 4., 1., -2., -4., -3., 2., 1.};
    const auto expected =
        Kokkos::View<double[3][4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 4U; ++j) {
            EXPECT_NEAR(derivative_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

Kokkos::View<double[4]> TestQuaternionInverse(const Kokkos::View<double[4]>::const_type& q) {
    auto q_inv = Kokkos::View<double[4]>("q_inv");
    Kokkos::parallel_for(
        "QuaternionInverse", 1, KOKKOS_LAMBDA(int) { math::QuaternionInverse(q, q_inv); }
    );
    return q_inv;
}

TEST(QuaternionTest, GetInverse) {
    const auto coeff = std::sqrt(30.);
    const auto q = Create1DView<4>({1. / coeff, 2. / coeff, 3. / coeff, 4. / coeff});

    const auto q_inv = TestQuaternionInverse(q);

    const auto q_inv_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q_inv);

    const auto expected_data = std::array{1. / coeff, -2. / coeff, -3. / coeff, -4. / coeff};
    const auto expected =
        Kokkos::View<double[4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 4U; ++i) {
        EXPECT_NEAR(q_inv_mirror(i), expected(i), 1.e-15);
    }
}

Kokkos::View<double[4]> TestQuaternionCompose(
    const Kokkos::View<double[4]>::const_type& q1, const Kokkos::View<double[4]>::const_type& q2
) {
    auto qn = Kokkos::View<double[4]>("qn");
    Kokkos::parallel_for(
        "QuaternionCompose", 1, KOKKOS_LAMBDA(int) { math::QuaternionCompose(q1, q2, qn); }
    );
    return qn;
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set1) {
    const auto q1 = Create1DView<4>({3., 1., -2., 1.});
    const auto q2 = Create1DView<4>({2., -1., 2., 3.});
    const auto qn = TestQuaternionCompose(q1, q2);

    const auto qn_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qn);

    constexpr auto expected_data = std::array{8., -9., -2., 11.};
    const auto expected =
        Kokkos::View<double[4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 4; ++i) {
        EXPECT_NEAR(qn_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set2) {
    auto q1 = Create1DView<4>({1., 2., 3., 4.});
    auto q2 = Create1DView<4>({5., 6., 7., 8.});
    const auto qn = TestQuaternionCompose(q1, q2);

    const auto qn_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qn);

    constexpr auto expected_data = std::array{-60., 12., 30., 24.};
    const auto expected =
        Kokkos::View<double[4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 4; ++i) {
        EXPECT_NEAR(qn_mirror(i), expected(i), 1.e-15);
    }
}

Kokkos::View<double[4]> TestRotationToQuaternion(const Kokkos::View<double[3]>::const_type& phi) {
    auto quaternion = Kokkos::View<double[4]>("quaternion");
    Kokkos::parallel_for(
        "RotationVectorToQuaternion", 1,
        KOKKOS_LAMBDA(int) { math::RotationVectorToQuaternion(phi, quaternion); }
    );
    return quaternion;
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set0) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    const auto phi = Create1DView<3>({M_PI / 2., 0., 0.});
    const auto quaternion = TestRotationToQuaternion(phi);

    const auto quaternion_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), quaternion);

    const auto expected_data = std::array{inv_sqrt2, inv_sqrt2, 0., 0.};
    const auto expected =
        Kokkos::View<double[4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 4U; ++i) {
        EXPECT_NEAR(quaternion_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set1) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    const auto phi = Create1DView<3>({0., M_PI / 2., 0.});
    const auto quaternion = TestRotationToQuaternion(phi);

    const auto quaternion_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), quaternion);

    const auto expected_data = std::array{inv_sqrt2, 0., inv_sqrt2, 0.};
    const auto expected =
        Kokkos::View<double[4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 4U; ++i) {
        EXPECT_NEAR(quaternion_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set2) {
    const auto inv_sqrt2 = 1. / std::numbers::sqrt2;
    const auto phi = Create1DView<3>({0., 0., M_PI / 2.});
    const auto quaternion = TestRotationToQuaternion(phi);

    const auto quaternion_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), quaternion);

    const auto expected_data = std::array{inv_sqrt2, 0., 0., inv_sqrt2};
    const auto expected =
        Kokkos::View<double[4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 4U; ++i) {
        EXPECT_NEAR(quaternion_mirror(i), expected(i), 1.e-15);
    }
}

void test_quaternion_to_rotation_vector_2() {
    const auto n = 100U;
    const auto dtheta = M_PI / static_cast<double>(n);
    for (auto i = 0U; i < n; ++i) {
        auto rot_vec = std::array{static_cast<double>(i) * dtheta, 0., 0.};
        auto phi = Create1DView<3>(rot_vec);
        auto phi2 = Create1DView<3>({0., 0., 0.});
        auto q = Create1DView<4>({0., 0., 0., 0.});
        Kokkos::parallel_for(
            "RotationVectorToQuaternion", 1,
            KOKKOS_LAMBDA(int) { math::RotationVectorToQuaternion(phi, q); }
        );
        Kokkos::parallel_for(
            "RotationVectorToQuaternion", 1,
            KOKKOS_LAMBDA(int) { math::QuaternionToRotationVector(q, phi2); }
        );

        const auto phi2_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi2);
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(phi2_mirror(j), rot_vec[j], 1.e-14);
        }
    }
}

TEST(QuaternionTest, QuaternionToRotationVector_1) {
    test_quaternion_to_rotation_vector_2();
}

TEST(QuaternionTest, QuaternionToRotationVector_2) {
    const auto n = 100U;
    const auto dtheta = M_PI / static_cast<double>(n);
    for (auto i = 0U; i < n; ++i) {
        auto rot_vec = std::array{static_cast<double>(i) * dtheta, 0., 0.};
        auto q = math::RotationVectorToQuaternion(rot_vec);
        auto rot_vec2 = math::QuaternionToRotationVector(q);
        ASSERT_NEAR(rot_vec2[0], rot_vec[0], 1e-14);
        ASSERT_NEAR(rot_vec2[1], rot_vec[1], 1e-14);
        ASSERT_NEAR(rot_vec2[2], rot_vec[2], 1e-14);
    }
}

TEST(QuaternionTest, CheckTangentTwistToQuaternion) {
    struct TestData {
        double twist;
        std::array<double, 3> tan;
        std::array<double, 4> q_exp;
    };
    for (const auto& td : std::vector<TestData>{
             {
                 45.,
                 {1., 0., 0.},
                 {0.92387953251128674, 0.38268343236508978, 0., 0.},
             },
             {
                 180.,
                 {1., 1., 0.},
                 {0., 0.92387953251128685, 0.38268343236508978, 0.},
             },
             {
                 45.,
                 {0., 0., 1.},
                 {0.65328148243818829, 0.27059805007309845, -0.65328148243818818, 0.27059805007309851
                 },
             },
         }) {
        const auto q_act = math::TangentTwistToQuaternion(td.tan, td.twist);
        ASSERT_NEAR(q_act[0], td.q_exp[0], 1e-14);
        ASSERT_NEAR(q_act[1], td.q_exp[1], 1e-14);
        ASSERT_NEAR(q_act[2], td.q_exp[2], 1e-14);
        ASSERT_NEAR(q_act[3], td.q_exp[3], 1e-14);
    }
}

TEST(QuaternionTest, IsIdentityQuaternion_ExactIdentity) {
    const auto identity_q = std::array{1., 0., 0., 0.};
    EXPECT_TRUE(math::IsIdentityQuaternion(identity_q));
}

TEST(QuaternionTest, IsIdentityQuaternion_WithinDefaultTolerance) {
    // 1e-13 is within default tolerance
    const auto near_identity_q = std::array{1. + 1e-13, 1e-13, -1e-13, 1e-13};
    EXPECT_TRUE(math::IsIdentityQuaternion(near_identity_q));
}

TEST(QuaternionTest, IsIdentityQuaternion_OutsideDefaultTolerance) {
    // 1e-11 is outside default tolerance
    const auto not_identity_q = std::array{1. + 1e-11, 0., 0., 0.};
    EXPECT_FALSE(math::IsIdentityQuaternion(not_identity_q));
}

TEST(QuaternionTest, IsIdentityQuaternion_WithCustomTolerance) {
    // 1e-10 is within custom tolerance
    const auto near_identity_q = std::array{1. + 1e-10, 1e-10, 0., 0.};
    EXPECT_TRUE(math::IsIdentityQuaternion(near_identity_q, 1e-9));
}

TEST(QuaternionTest, IsIdentityQuaternion_NonIdentityQuaternions) {
    // 90 degree rotation about X axis
    const auto rotation_x = math::RotationVectorToQuaternion({M_PI / 2., 0., 0.});
    EXPECT_FALSE(math::IsIdentityQuaternion(rotation_x));

    // 90 degree rotation about Y axis
    const auto rotation_y = math::RotationVectorToQuaternion({0., M_PI / 2., 0.});
    EXPECT_FALSE(math::IsIdentityQuaternion(rotation_y));

    // 90 degree rotation about Z axis
    const auto rotation_z = math::RotationVectorToQuaternion({0., 0., M_PI / 2.});
    EXPECT_FALSE(math::IsIdentityQuaternion(rotation_z));

    // Arbitrary quaternion
    const auto arbitrary = std::array{0.5, 0.5, 0.5, 0.5};
    EXPECT_FALSE(math::IsIdentityQuaternion(arbitrary));
}

}  // namespace openturbine::tests
