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

}  // namespace openturbine::tests
