#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/math/vector_operations.hpp"

namespace openturbine::tests {

template <unsigned size>
Kokkos::View<double[size]> Create1DView(const std::array<double, size>& input) {
    auto view = Kokkos::View<double[size]>("view");
    auto view_host = typename Kokkos::View<double[size], Kokkos::HostSpace>::const_type(input.data());
    Kokkos::deep_copy(view, view_host);
    return view;
}

Kokkos::View<double[3][3]> TestVecTilde(const Kokkos::View<double[3]>& v) {
    auto m = Kokkos::View<double[3][3]>("m");
    Kokkos::parallel_for("VecTilde", 1, KOKKOS_LAMBDA(int) { VecTilde(v, m); });
    return m;
}

TEST(VectorTest, VecTilde) {
    auto v = Create1DView<3>({1., 2., 3.});
    const auto m = TestVecTilde(v);

    const auto m_mirror = Kokkos::create_mirror(m);
    Kokkos::deep_copy(m_mirror, m);

    constexpr auto expected_data = std::array{0., -3., 2., 3., 0., -1., -2., 1., 0.};
    const auto expected = Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type(expected_data.data());

    for(auto i = 0U; i < 3U; ++i) {
        for(auto j = 0U; j < 3U; ++j) {
            EXPECT_NEAR(m_mirror(i, j), expected(i, j), 1.e-15);
        }
    }
}

TEST(VectorTest, CrossProduct_Set1) {
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

void test_DotProduct_View() {
    auto a = Create1DView<3>({1., 2., 3.});
    auto b = Create1DView<3>({4., 5., 6.});
    auto c = 0.;
    Kokkos::parallel_reduce(
        "DotProduct_View", 1, KOKKOS_LAMBDA(int, double& result) { result = DotProduct(a, b); }, c
    );
    ASSERT_EQ(c, 32.);
}

TEST(VectorTest, DotProduct_View) {
    test_DotProduct_View();
}

TEST(VectorTest, DotProduct_Array) {
    auto a = std::array<double, 3>{1., 2., 3.};
    auto b = std::array<double, 3>{4., 5., 6.};
    auto c = DotProduct(a, b);
    ASSERT_EQ(c, 32);
}

TEST(VectorTest, UnitVector_Set1) {
    auto a = std::array<double, 3>{5., 0., 0.};
    auto b = UnitVector(a);
    ASSERT_EQ(b[0], 1.);
    ASSERT_EQ(b[1], 0.);
    ASSERT_EQ(b[2], 0.);
}

TEST(VectorTest, UnitVector_Set2) {
    auto a = std::array<double, 3>{3., 4., 0.};
    auto b = UnitVector(a);
    ASSERT_EQ(b[0], 0.6);
    ASSERT_EQ(b[1], 0.8);
    ASSERT_EQ(b[2], 0.);
}

TEST(VectorTest, VectorTest_UnitVector_Set3_Test) {
    auto a = std::array<double, 3>{0., 0., 0.};
    EXPECT_THROW(UnitVector(a), std::invalid_argument);
}

}  // namespace openturbine::tests
