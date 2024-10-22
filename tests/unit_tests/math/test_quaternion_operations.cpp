#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/math/quaternion_operations.hpp"

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
        KOKKOS_LAMBDA(int) { QuaternionToRotationMatrix(q, R_from_q); }
    );
    return R_from_q;
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutXAxis) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    const auto rotation_x_axis = Create1DView<4>(std::array{inv_sqrt2, inv_sqrt2, 0., 0.});

    const auto R_from_q = TestQuaternionToRotationMatrix(rotation_x_axis);

    const auto R_from_q_mirror = Kokkos::create_mirror(R_from_q);
    Kokkos::deep_copy(R_from_q_mirror, R_from_q);
    constexpr auto expected_data = std::array{1., 0., 0., 0., 0., -1., 0., 1., 0.};
    const auto expected = Kokkos::View<const double[3][3], Kokkos::HostSpace>(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(R_from_q_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutYAxis) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    const auto rotation_x_axis = Create1DView<4>(std::array{inv_sqrt2, 0., inv_sqrt2, 0.});

    const auto R_from_q = TestQuaternionToRotationMatrix(rotation_x_axis);

    const auto R_from_q_mirror = Kokkos::create_mirror(R_from_q);
    Kokkos::deep_copy(R_from_q_mirror, R_from_q);
    constexpr auto expected_data = std::array{0., 0., 1., 0., 1., 0., -1., 0., 0.};
    const auto expected = Kokkos::View<const double[3][3], Kokkos::HostSpace>(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(R_from_q_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutZAxis) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    const auto rotation_x_axis = Create1DView<4>(std::array{inv_sqrt2, 0., 0., inv_sqrt2});

    const auto R_from_q = TestQuaternionToRotationMatrix(rotation_x_axis);

    const auto R_from_q_mirror = Kokkos::create_mirror(R_from_q);
    Kokkos::deep_copy(R_from_q_mirror, R_from_q);
    constexpr auto expected_data = std::array{0., -1., 0., 1., 0., 0., 0., 0., 1.};
    const auto expected = Kokkos::View<const double[3][3], Kokkos::HostSpace>(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(R_from_q_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

Kokkos::View<double[3]> TestRotateVectorByQuaternion(
    const Kokkos::View<double[4]>::const_type& q, const Kokkos::View<double[3]>::const_type& v
) {
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    Kokkos::parallel_for(
        "RotateVectorBoyQuaternion", 1, KOKKOS_LAMBDA(int) { RotateVectorByQuaternion(q, v, v_rot); }
    );
    return v_rot;
}

TEST(QuaternionTest, RotateYAxisByIdentity) {
    auto rotation_identity = Create1DView<4>({1., 0., 0., 0.});
    auto y_axis = Create1DView<3>({0., 1., 0.});

    const auto v_rot = TestRotateVectorByQuaternion(rotation_identity, y_axis);

    const auto v_rot_mirror = Kokkos::create_mirror(v_rot);
    Kokkos::deep_copy(v_rot_mirror, v_rot);

    constexpr auto expected_data = std::array{0., 1., 0.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotateXAxis90DegreesAboutYAxis) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    auto rotation_y_axis = Create1DView<4>({inv_sqrt2, 0., inv_sqrt2, 0.});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    const auto v_rot = TestRotateVectorByQuaternion(rotation_y_axis, x_axis);

    const auto v_rot_mirror = Kokkos::create_mirror(v_rot);
    Kokkos::deep_copy(v_rot_mirror, v_rot);

    constexpr auto expected_data = std::array{0., 0., -1.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotateZAxis90DegreesAboutXAxis) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    auto rotation_x_axis = Create1DView<4>({inv_sqrt2, inv_sqrt2, 0., 0.});
    auto z_axis = Create1DView<3>({0., 0., 1.});
    const auto v_rot = TestRotateVectorByQuaternion(rotation_x_axis, z_axis);

    const auto v_rot_mirror = Kokkos::create_mirror(v_rot);
    Kokkos::deep_copy(v_rot_mirror, v_rot);

    constexpr auto expected_data = std::array{0., -1., 0.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotateXAxis45DegreesAboutZAxis) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    const auto cos_pi_8 = std::cos(M_PI / 8.);
    const auto sin_pi_8 = std::sin(M_PI / 8.);
    auto rotation_z_axis = Create1DView<4>({cos_pi_8, 0., 0., sin_pi_8});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    const auto v_rot = TestRotateVectorByQuaternion(rotation_z_axis, x_axis);

    const auto v_rot_mirror = Kokkos::create_mirror(v_rot);
    Kokkos::deep_copy(v_rot_mirror, v_rot);

    const auto expected_data = std::array{inv_sqrt2, inv_sqrt2, 0.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_rot_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotateXAxisNeg45DegreesAboutZAxis) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    const auto cos_pi_8 = std::cos(M_PI / 8.);
    const auto sin_pi_8 = std::sin(M_PI / 8.);
    auto rotation_z_axis = Create1DView<4>({cos_pi_8, 0., 0., -sin_pi_8});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    const auto v_rot = TestRotateVectorByQuaternion(rotation_z_axis, x_axis);

    const auto v_rot_mirror = Kokkos::create_mirror(v_rot);
    Kokkos::deep_copy(v_rot_mirror, v_rot);

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
        "QuaternionDerivative", 1, KOKKOS_LAMBDA(int) { QuaternionDerivative(q, m); }
    );
    return m;
}

TEST(QuaternionTest, QuaternionDerivative) {
    const auto q = Create1DView<4>({1., 2., 3., 4.});
    const auto derivative = TestQuaternionDerivative(q);

    const auto derivative_mirror = Kokkos::create_mirror(derivative);
    Kokkos::deep_copy(derivative_mirror, derivative);

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
        "QuaternionInverse", 1, KOKKOS_LAMBDA(int) { QuaternionInverse(q, q_inv); }
    );
    return q_inv;
}

TEST(QuaternionTest, GetInverse) {
    const auto coeff = std::sqrt(30.);
    const auto q = Create1DView<4>({1. / coeff, 2. / coeff, 3. / coeff, 4. / coeff});

    const auto q_inv = TestQuaternionInverse(q);

    const auto q_inv_mirror = Kokkos::create_mirror(q_inv);
    Kokkos::deep_copy(q_inv_mirror, q_inv);

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
        "QuaternionCompose", 1, KOKKOS_LAMBDA(int) { QuaternionCompose(q1, q2, qn); }
    );
    return qn;
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set1) {
    const auto q1 = Create1DView<4>({3., 1., -2., 1.});
    const auto q2 = Create1DView<4>({2., -1., 2., 3.});
    const auto qn = TestQuaternionCompose(q1, q2);

    const auto qn_mirror = Kokkos::create_mirror(qn);
    Kokkos::deep_copy(qn_mirror, qn);

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

    const auto qn_mirror = Kokkos::create_mirror(qn);
    Kokkos::deep_copy(qn_mirror, qn);

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
        KOKKOS_LAMBDA(int) { RotationVectorToQuaternion(phi, quaternion); }
    );
    return quaternion;
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set0) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    const auto phi = Create1DView<3>({M_PI / 2., 0., 0.});
    const auto quaternion = TestRotationToQuaternion(phi);

    const auto quaternion_mirror = Kokkos::create_mirror(quaternion);
    Kokkos::deep_copy(quaternion_mirror, quaternion);

    const auto expected_data = std::array{inv_sqrt2, inv_sqrt2, 0., 0.};
    const auto expected =
        Kokkos::View<double[4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 4U; ++i) {
        EXPECT_NEAR(quaternion_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set1) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    const auto phi = Create1DView<3>({0., M_PI / 2., 0.});
    const auto quaternion = TestRotationToQuaternion(phi);

    const auto quaternion_mirror = Kokkos::create_mirror(quaternion);
    Kokkos::deep_copy(quaternion_mirror, quaternion);

    const auto expected_data = std::array{inv_sqrt2, 0., inv_sqrt2, 0.};
    const auto expected =
        Kokkos::View<double[4], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 4U; ++i) {
        EXPECT_NEAR(quaternion_mirror(i), expected(i), 1.e-15);
    }
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set2) {
    const auto inv_sqrt2 = 1. / std::sqrt(2.);
    const auto phi = Create1DView<3>({0., 0., M_PI / 2.});
    const auto quaternion = TestRotationToQuaternion(phi);

    const auto quaternion_mirror = Kokkos::create_mirror(quaternion);
    Kokkos::deep_copy(quaternion_mirror, quaternion);

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
    for (size_t i = 0; i < n; ++i) {
        Array_3 rot_vec{static_cast<double>(i) * dtheta, 0., 0.};
        auto phi = Create1DView<3>(rot_vec);
        auto phi2 = Create1DView<3>({0., 0., 0.});
        auto q = Create1DView<4>({0., 0., 0., 0.});
        Kokkos::parallel_for(
            "RotationVectorToQuaternion", 1,
            KOKKOS_LAMBDA(int) { RotationVectorToQuaternion(phi, q); }
        );
        Kokkos::parallel_for(
            "RotationVectorToQuaternion", 1,
            KOKKOS_LAMBDA(int) { QuaternionToRotationVector(q, phi2); }
        );

        const auto phi2_mirror = Kokkos::create_mirror(phi2);
        Kokkos::deep_copy(phi2_mirror, phi2);
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(phi2_mirror(j), rot_vec[j], 1.e-15);
        }
    }
}

TEST(QuaternionTest, QuaternionToRotationVector_1) {
    test_quaternion_to_rotation_vector_2();
}

TEST(QuaternionTest, QuaternionToRotationVector_2) {
    const auto n = 100U;
    const auto dtheta = M_PI / static_cast<double>(n);
    for (size_t i = 0; i < n; ++i) {
        Array_3 rot_vec{static_cast<double>(i) * dtheta, 0., 0.};
        auto q = RotationVectorToQuaternion(rot_vec);
        auto rot_vec2 = QuaternionToRotationVector(q);
        ASSERT_NEAR(rot_vec2[0], rot_vec[0], 1e-7);
        ASSERT_NEAR(rot_vec2[1], rot_vec[1], 1e-7);
        ASSERT_NEAR(rot_vec2[2], rot_vec[2], 1e-7);
    }
}

}  // namespace openturbine::tests
