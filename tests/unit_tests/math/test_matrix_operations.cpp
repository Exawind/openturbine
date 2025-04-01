#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "math/matrix_operations.hpp"

namespace openturbine::tests {

template <size_t rows, size_t cols>
Kokkos::View<double[rows][cols]> Create2DView(const std::array<double, rows * cols>& input) {
    auto view = Kokkos::View<double[rows][cols]>("view");
    auto view_host = Kokkos::View<const double[rows][cols], Kokkos::HostSpace>(input.data());
    auto view_mirror = Kokkos::create_mirror(view);
    Kokkos::deep_copy(view_mirror, view_host);
    Kokkos::deep_copy(view, view_mirror);
    return view;
}

inline void test_AX_Matrix() {
    const auto A = Create2DView<3, 3>({1., 2., 3., 4., 5., 6., 7., 8., 9.});
    const auto out = Kokkos::View<double[3][3]>("out");
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) { AX_Matrix(A, out); });
    const auto out_mirror = Kokkos::create_mirror(out);
    Kokkos::deep_copy(out_mirror, out);

    constexpr auto expected_data = std::array{7., -1., -1.5, -2., 5., -3., -3.5, -4., 3.};
    const auto expected =
        Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(out_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

TEST(MatrixTest, AX_Matrix) {
    test_AX_Matrix();
}

Kokkos::View<double[3]> TestAxialVectorOfMatrix(const Kokkos::View<const double[3][3]>& m) {
    auto v = Kokkos::View<double[3]>("v");
    Kokkos::parallel_for(
        "AxialVectorOfMatrix", 1, KOKKOS_LAMBDA(int) { AxialVectorOfMatrix(m, v); }
    );
    return v;
}

TEST(MatrixTest, AxialVectorOfMatrix) {
    const auto m = Create2DView<3, 3>({0., -1., 0., 1., 0., 0., 0., 0., 0.});
    const auto v = TestAxialVectorOfMatrix(m);

    const auto v_mirror = Kokkos::create_mirror(v);
    Kokkos::deep_copy(v_mirror, v);

    constexpr auto expected_data = std::array{0., 0., 1.};
    const auto expected =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(expected_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(v_mirror(i), expected(i), 1.e-15);
    }
}

TEST(MatrixTest, RotateMatrix6_1) {
    const Array_6x6 m{{
        {1., 2., 3., 4., 5., 6.},
        {7., 8., 9., 10., 11., 12.},
        {13., 14., 15., 16., 17., 18.},
        {19., 20., 21., 22., 23., 24.},
        {25., 26., 27., 28., 29., 30.},
        {31., 32., 33., 34., 35., 36.},
    }};
    const Array_6x6 m_exp{{
        {1., 2., 3., 4., 5., 6.},
        {7., 8., 9., 10., 11., 12.},
        {13., 14., 15., 16., 17., 18.},
        {19., 20., 21., 22., 23., 24.},
        {25., 26., 27., 28., 29., 30.},
        {31., 32., 33., 34., 35., 36.},
    }};
    const Array_4 q{1., 0., 0., 0.};
    const auto m_act = RotateMatrix6(m, q);
    for (auto i = 0U; i < 6U; ++i) {
        for (auto j = 0U; j < 6U; ++j) {
            EXPECT_DOUBLE_EQ(m_act[i][j], m_exp[i][j]);
        }
    }
}

TEST(MatrixTest, RotateMatrix6_2) {
    const Array_6x6 m{{
        {1., 2., 3., 4., 5., 6.},
        {7., 8., 9., 10., 11., 12.},
        {13., 14., 15., 16., 17., 18.},
        {19., 20., 21., 22., 23., 24.},
        {25., 26., 27., 28., 29., 30.},
        {31., 32., 33., 34., 35., 36.},
    }};
    const Array_6x6 m_exp{{
        {1, -2, -3, 4, -5, -6},
        {-7, 8, 9, -10, 11, 12},
        {-13, 14, 15, -16, 17, 18},
        {19, -20, -21, 22, -23, -24},
        {-25, 26, 27, -28, 29, 30},
        {-31, 32, 33, -34, 35, 36},
    }};
    const Array_4 q{0., 1., 0., 0.};
    const auto m_act = RotateMatrix6(m, q);
    for (auto i = 0U; i < 6U; ++i) {
        for (auto j = 0U; j < 6U; ++j) {
            EXPECT_DOUBLE_EQ(m_act[i][j], m_exp[i][j]);
        }
    }
}

TEST(MatrixTest, RotateMatrix6_3) {
    const Array_6x6 m{{
        {1., 2., 3., 4., 5., 6.},
        {7., 8., 9., 10., 11., 12.},
        {13., 14., 15., 16., 17., 18.},
        {19., 20., 21., 22., 23., 24.},
        {25., 26., 27., 28., 29., 30.},
        {31., 32., 33., 34., 35., 36.},
    }};
    const Array_6x6 m_exp{{
        {16, 11.313708498984761, 1.9999999999999982, 22.000000000000004, 15.556349186104047,
         1.9999999999999987},
        {11.313708498984759, 8, 1.4142135623730945, 15.556349186104045, 11, 1.414213562373094},
        {11.999999999999998, 8.4852813742385695, -1.0638139115010993E-15, 11.999999999999998,
         8.4852813742385695, -2.3198878784481192E-15},
        {52.000000000000007, 36.76955262170047, 1.999999999999996, 58, 41.012193308819761,
         1.9999999999999949},
        {36.76955262170047, 26, 1.4142135623730927, 41.012193308819754, 29, 1.4142135623730931},
        {11.999999999999996, 8.4852813742385678, -5.4353103904786874E-16, 11.999999999999995,
         8.4852813742385678, -1.7996050059948887E-15},

    }};
    const Array_4 q{RotationVectorToQuaternion({0., M_PI / 4., 0.})};
    const auto m_act = RotateMatrix6(m, q);
    for (auto i = 0U; i < 6U; ++i) {
        for (auto j = 0U; j < 6U; ++j) {
            EXPECT_DOUBLE_EQ(m_act[i][j], m_exp[i][j]);
        }
    }
}

}  // namespace openturbine::tests
