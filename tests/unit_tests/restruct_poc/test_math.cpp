#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"

namespace openturbine::restruct_poc::tests {

template <unsigned size>
auto Create1DView(const std::array<double, size>& input) {
    auto view = Kokkos::View<double[size]>("view");
    auto view_host =
        Kokkos::View<const double[size], Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            input.data()
        );
    Kokkos::deep_copy(view, view_host);
    return view;
}

template <unsigned rows, unsigned cols>
auto Create2DView(const std::array<std::array<double, rows>, cols>& input) {
    auto view = Kokkos::View<double[rows][cols]>("view");
    auto view_host = Kokkos::create_mirror(view);
    for (auto i = 0U; i < rows; i++) {
        for (auto j = 0U; j < cols; j++) {
            view_host(i, j) = input[i][j];
        }
    }
    Kokkos::deep_copy(view, view_host);
    return view;
}

void TestConversion(
    const Kokkos::View<const double[4]>& q, const std::vector<std::vector<double>>& expected
) {
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    Kokkos::parallel_for(
        "QuaternionToRotationMatrix", 1,
        KOKKOS_LAMBDA(int) { QuaternionToRotationMatrix(q, R_from_q); }
    );
    expect_kokkos_view_2D_equal(R_from_q, expected);
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutXAxis) {
    auto rotation_x_axis = Create1DView<4>({
        0.707107,
        0.707107,
        0.,
        0.,
    });
    TestConversion(rotation_x_axis, {{1., 0., 0.}, {0., 0., -1.}, {0., 1., 0.}});
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutYAxis) {
    auto rotation_y_axis = Create1DView<4>({0.707107, 0., 0.707107, 0.});
    ;
    TestConversion(rotation_y_axis, {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., 0.}});
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_90DegreeRotationAboutZAxis) {
    auto rotation_z_axis = Create1DView<4>({0.707107, 0., 0., 0.707107});
    ;
    TestConversion(rotation_z_axis, {{0., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}});
}

void TestRotation(
    const Kokkos::View<const double[4]>& q, const Kokkos::View<const double[3]>& v,
    const std::vector<double>& exact
) {
    auto v_rot = Kokkos::View<double[3]>("v_rot");
    Kokkos::parallel_for(
        "RotateVectorBoyQuaternion", 1, KOKKOS_LAMBDA(int) { RotateVectorByQuaternion(q, v, v_rot); }
    );
    expect_kokkos_view_1D_equal(v_rot, exact);
}

TEST(QuaternionTest, RotateYAxisByIdentity) {
    auto rotation_identity = Create1DView<4>({1., 0., 0., 0.});
    auto y_axis = Create1DView<3>({0., 1., 0.});
    TestRotation(rotation_identity, y_axis, {0., 1., 0.});
}

TEST(QuaternionTest, RotateXAxis90DegreesAboutYAxis) {
    auto rotation_y_axis = Create1DView<4>({0.707107, 0., 0.707107, 0.});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    TestRotation(rotation_y_axis, x_axis, {0., 0., -1.});
}

TEST(QuaternionTest, RotateZAxis90DegreesAboutXAxis) {
    auto rotation_x_axis = Create1DView<4>({0.707107, 0.707107, 0., 0.});
    auto z_axis = Create1DView<3>({0., 0., 1.});
    TestRotation(rotation_x_axis, z_axis, {0., -1., 0.});
}

TEST(QuaternionTest, RotateXAxis45DegreesAboutZAxis) {
    auto rotation_z_axis = Create1DView<4>({0.92388, 0., 0., 0.382683});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    TestRotation(rotation_z_axis, x_axis, {0.707107, 0.707107, 0.});
}

TEST(QuaternionTest, RotateXAxisNeg45DegreesAboutZAxis) {
    auto rotation_z_axis = Create1DView<4>({0.92388, 0., 0., -0.382683});
    auto x_axis = Create1DView<3>({1., 0., 0.});
    TestRotation(rotation_z_axis, x_axis, {0.707107, -0.707107, 0.});
}

void TestDerivative(
    const Kokkos::View<const double[4]>& q, const std::vector<std::vector<double>>& exact
) {
    auto m = Kokkos::View<double[3][4]>("m");
    Kokkos::parallel_for(
        "QuaternionDerivative", 1, KOKKOS_LAMBDA(int) { QuaternionDerivative(q, m); }
    );
    expect_kokkos_view_2D_equal(m, exact);
}

TEST(QuaternionTest, QuaternionDerivative) {
    auto q = Create1DView<4>({1., 2., 3., 4.});
    TestDerivative(q, {{-2., 1., -4., 3.}, {-3., 4., 1., -2.}, {-4., -3., 2., 1.}});
}

void TestInverse(const Kokkos::View<const double[4]>& q, const std::vector<double>& exact) {
    auto q_inv = Kokkos::View<double[4]>("q_inv");
    Kokkos::parallel_for(
        "QuaternionInverse", 1, KOKKOS_LAMBDA(int) { QuaternionInverse(q, q_inv); }
    );
    expect_kokkos_view_1D_equal(q_inv, exact);
}

TEST(QuaternionTest, GetInverse) {
    const auto coeff = std::sqrt(30.);
    auto q = Create1DView<4>({1. / coeff, 2. / coeff, 3. / coeff, 4. / coeff});
    TestInverse(q, {1. / coeff, -2. / coeff, -3. / coeff, -4. / coeff});
}

void TestCompose(
    const Kokkos::View<const double[4]>& q1, const Kokkos::View<const double[4]>& q2,
    const std::vector<double>& exact
) {
    auto qn = Kokkos::View<double[4]>("qn");
    Kokkos::parallel_for(
        "QuaternionCompose", 1, KOKKOS_LAMBDA(int) { QuaternionCompose(q1, q2, qn); }
    );
    expect_kokkos_view_1D_equal(qn, exact);
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set1) {
    auto q1 = Create1DView<4>({3., 1., -2., 1.});
    auto q2 = Create1DView<4>({2., -1., 2., 3.});
    TestCompose(q1, q2, {8., -9., -2., 11.});
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set2) {
    auto q1 = Create1DView<4>({1., 2., 3., 4.});
    auto q2 = Create1DView<4>({5., 6., 7., 8.});
    TestCompose(q1, q2, {-60., 12., 30., 24.});
}

auto Create2DView(const std::array<double, 9>& input) {
    auto m = Kokkos::View<double[3][3]>("m");
    auto m_host = Kokkos::create_mirror(m);
    auto input_view =
        Kokkos::View<const double[3][3], Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            input.data()
        );
    Kokkos::deep_copy(m_host, input_view);
    Kokkos::deep_copy(m, m_host);
    return m;
}

void TestAxialVector(const Kokkos::View<const double[3][3]>& m, const std::vector<double>& exact) {
    auto v = Kokkos::View<double[3]>("v");
    Kokkos::parallel_for(
        "AxialVectorOfMatrix", 1, KOKKOS_LAMBDA(int) { AxialVectorOfMatrix(m, v); }
    );
    expect_kokkos_view_1D_equal(v, exact);
}

TEST(VectorTest, AxialVectorOfMatrix) {
    auto m = Create2DView({0., -1., 0., 1., 0., 0., 0., 0., 0.});
    TestAxialVector(m, {0., 0., 1.});
}

void TestRotationToQuaternion(
    const Kokkos::View<const double[4]>& phi, const std::vector<double>& exact
) {
    auto quaternion = Kokkos::View<double[4]>("quaternion");
    Kokkos::parallel_for(
        "RotationVectorToQuaternion", 1,
        KOKKOS_LAMBDA(int) { RotationVectorToQuaternion(phi, quaternion); }
    );
    expect_kokkos_view_1D_equal(quaternion, exact);
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set1) {
    auto phi = Create1DView<4>({1., 2., 3.});
    TestRotationToQuaternion(phi, {-0.295551, 0.255322, 0.510644, 0.765966});
}

TEST(QuaternionTest, RotationVectorToQuaternion_Set2) {
    auto phi = Create1DView<4>({0., 0., 1.570796});
    TestRotationToQuaternion(phi, {0.707107, 0., 0., 0.707107});
}

void TestVecTilde(const Kokkos::View<double[3]>& v, const std::vector<std::vector<double>>& exact) {
    auto m = Kokkos::View<double[3][3]>("m");
    Kokkos::parallel_for(
        "VecTilde", 1, KOKKOS_LAMBDA(int) { VecTilde(v, m); }
    );
    expect_kokkos_view_2D_equal(m, exact);
}

TEST(VectorTest, VecTilde) {
    auto v = Create1DView<3>({1., 2., 3.});
    TestVecTilde(v, {{0., -3., 2.}, {3., 0., -1.}, {-2., 1., 0.}});
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

inline void test_AX_Matrix() {
    auto A = Create2DView<3, 3>({{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}});
    auto out = Create2DView<3, 3>({{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}});
    Kokkos::parallel_for(
        1, KOKKOS_LAMBDA(int) { AX_Matrix(A, out); }
    );
    auto tmp = kokkos_view_2D_to_vector(out);
    expect_kokkos_view_2D_equal(
        out,
        {
            {7, -1, -1.5},
            {-2, 5, -3},
            {-3.5, -4, 3},
        }
    );
}

TEST(MatrixTest, AX_Matrix) {
    test_AX_Matrix();
}

}  // namespace openturbine::restruct_poc::tests
