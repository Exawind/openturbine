#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_integrate_matrix.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/integrate_residual_vector.hpp"
#include "src/restruct_poc/types.hpp"
#include "tests/unit_tests/restruct_poc/test_utilities.hpp"

namespace openturbine::tests {

void TestIntegrateResidualVector_1Element1Node_AllZeros() {
    constexpr auto num_elements = size_t{1U};
    constexpr auto num_nodes = size_t{1U};
    const auto node_state_indices = get_node_state_indices<num_elements, num_nodes>();

    const auto node_FE = Kokkos::View<double[num_elements * num_nodes][6]>("node_FE");
    const auto node_FI = Kokkos::View<double[num_elements * num_nodes][6]>("node_FI");
    const auto node_FG = Kokkos::View<double[num_elements * num_nodes][6]>("node_FG");
    const auto node_FX = Kokkos::View<double[num_elements * num_nodes][6]>("node_FX");

    auto residual_vector = Kokkos::View<double[num_elements * num_nodes * 6]>("residual_vector");

    Kokkos::parallel_for(
        num_elements * num_nodes,
        IntegrateResidualVector{
            node_state_indices, node_FE, node_FI, node_FG, node_FX, residual_vector}
    );

    expect_kokkos_view_1D_equal(residual_vector, {0., 0., 0., 0., 0., 0.});
}

TEST(IntegrateResidualVector, OneElementOneNode_AllZeros) {
    TestIntegrateResidualVector_1Element1Node_AllZeros();
}

template <size_t n_elem, size_t n_nodes>
auto get_node_vector(
    std::string_view name, const std::array<double, n_elem * n_nodes * 6>& vector_data
) {
    using VectorView = Kokkos::View<double[n_elem * n_nodes][6]>;
    using HostVectorView = Kokkos::View<const double[n_elem * n_nodes][6], Kokkos::HostSpace>;
    auto shape = VectorView(std::string{name});
    const auto host_shape = Kokkos::create_mirror(shape);
    const auto shape_data_view = HostVectorView(vector_data.data());
    Kokkos::deep_copy(host_shape, shape_data_view);
    Kokkos::deep_copy(shape, host_shape);
    return shape;
}

template <size_t n_elem, size_t n_nodes>
auto get_node_FE(const std::array<double, n_elem * n_nodes * 6>& vector_data) {
    return get_node_vector<n_elem, n_nodes>("node_FE", vector_data);
}

template <size_t n_elem, size_t n_nodes>
auto get_node_FI(const std::array<double, n_elem * n_nodes * 6>& vector_data) {
    return get_node_vector<n_elem, n_nodes>("node_FI", vector_data);
}

template <size_t n_elem, size_t n_nodes>
auto get_node_FG(const std::array<double, n_elem * n_nodes * 6>& vector_data) {
    return get_node_vector<n_elem, n_nodes>("node_FG", vector_data);
}

template <size_t n_elem, size_t n_nodes>
auto get_node_FX(const std::array<double, n_elem * n_nodes * 6>& vector_data) {
    return get_node_vector<n_elem, n_nodes>("node_FX", vector_data);
}

void TestIntegrateResidualVector_1Element1Node_node_FE() {
    constexpr auto num_elements = size_t{1U};
    constexpr auto num_nodes = size_t{1U};
    const auto node_state_indices = get_node_state_indices<num_elements, num_nodes>();

    const auto node_FE = get_node_FE<num_elements, num_nodes>({1., 2., 3., 4., 5., 6.});
    const auto node_FI = Kokkos::View<double[num_elements * num_nodes][6]>("node_FI");
    const auto node_FG = Kokkos::View<double[num_elements * num_nodes][6]>("node_FG");
    const auto node_FX = Kokkos::View<double[num_elements * num_nodes][6]>("node_FX");

    auto residual_vector = Kokkos::View<double[num_elements * num_nodes * 6]>("residual_vector");

    Kokkos::parallel_for(
        num_elements * num_nodes,
        IntegrateResidualVector{
            node_state_indices, node_FE, node_FI, node_FG, node_FX, residual_vector}
    );

    expect_kokkos_view_1D_equal(residual_vector, {1., 2., 3., 4., 5., 6.});
}

TEST(IntegrateResidualVector, OneElementOneNode_node_FE) {
    TestIntegrateResidualVector_1Element1Node_node_FE();
}

void TestIntegrateResidualVector_1Element1Node_node_FI() {
    constexpr auto num_elements = size_t{1U};
    constexpr auto num_nodes = size_t{1U};
    const auto node_state_indices = get_node_state_indices<num_elements, num_nodes>();

    const auto node_FE = Kokkos::View<double[num_elements * num_nodes][6]>("node_FE");
    const auto node_FI = get_node_FI<num_elements, num_nodes>({1., 2., 3., 4., 5., 6.});
    const auto node_FG = Kokkos::View<double[num_elements * num_nodes][6]>("node_FG");
    const auto node_FX = Kokkos::View<double[num_elements * num_nodes][6]>("node_FX");

    auto residual_vector = Kokkos::View<double[num_elements * num_nodes * 6]>("residual_vector");

    Kokkos::parallel_for(
        num_elements * num_nodes,
        IntegrateResidualVector{
            node_state_indices, node_FE, node_FI, node_FG, node_FX, residual_vector}
    );

    expect_kokkos_view_1D_equal(residual_vector, {1., 2., 3., 4., 5., 6.});
}

TEST(IntegrateResidualVector, OneElementOneNode_node_FI) {
    TestIntegrateResidualVector_1Element1Node_node_FI();
}

void TestIntegrateResidualVector_1Element1Node_node_FG() {
    constexpr auto num_elements = size_t{1U};
    constexpr auto num_nodes = size_t{1U};
    const auto node_state_indices = get_node_state_indices<num_elements, num_nodes>();

    const auto node_FE = Kokkos::View<double[num_elements * num_nodes][6]>("node_FE");
    const auto node_FI = Kokkos::View<double[num_elements * num_nodes][6]>("node_FI");
    const auto node_FG = get_node_FG<num_elements, num_nodes>({1., 2., 3., 4., 5., 6.});
    const auto node_FX = Kokkos::View<double[num_elements * num_nodes][6]>("node_FX");

    auto residual_vector = Kokkos::View<double[num_elements * num_nodes * 6]>("residual_vector");

    Kokkos::parallel_for(
        num_elements * num_nodes,
        IntegrateResidualVector{
            node_state_indices, node_FE, node_FI, node_FG, node_FX, residual_vector}
    );

    expect_kokkos_view_1D_equal(residual_vector, {-1., -2., -3., -4., -5., -6.});
}

TEST(IntegrateResidualVector, OneElementOneNode_node_FG) {
    TestIntegrateResidualVector_1Element1Node_node_FG();
}

void TestIntegrateResidualVector_1Element1Node_node_FX() {
    constexpr auto num_elements = size_t{1U};
    constexpr auto num_nodes = size_t{1U};
    const auto node_state_indices = get_node_state_indices<num_elements, num_nodes>();

    const auto node_FE = Kokkos::View<double[num_elements * num_nodes][6]>("node_FE");
    const auto node_FI = Kokkos::View<double[num_elements * num_nodes][6]>("node_FI");
    const auto node_FG = Kokkos::View<double[num_elements * num_nodes][6]>("node_FG");
    const auto node_FX = get_node_FX<num_elements, num_nodes>({1., 2., 3., 4., 5., 6.});

    auto residual_vector = Kokkos::View<double[num_elements * num_nodes * 6]>("residual_vector");

    Kokkos::parallel_for(
        num_elements * num_nodes,
        IntegrateResidualVector{
            node_state_indices, node_FE, node_FI, node_FG, node_FX, residual_vector}
    );

    expect_kokkos_view_1D_equal(residual_vector, {-1., -2., -3., -4., -5., -6.});
}

TEST(IntegrateResidualVector, OneElementOneNode_node_FX) {
    TestIntegrateResidualVector_1Element1Node_node_FX();
}

void TestIntegrateResidualVector_1Element1Node_Sum() {
    constexpr auto num_elements = size_t{1U};
    constexpr auto num_nodes = size_t{1U};
    const auto node_state_indices = get_node_state_indices<num_elements, num_nodes>();

    const auto node_FE = get_node_FE<num_elements, num_nodes>({1., 2., 3., 4., 5., 6.});
    const auto node_FI = get_node_FI<num_elements, num_nodes>({1., 2., 3., 4., 5., 6.});
    const auto node_FG = get_node_FG<num_elements, num_nodes>({1., 2., 3., 4., 5., 6.});
    const auto node_FX = get_node_FX<num_elements, num_nodes>({1., 2., 3., 4., 5., 6.});

    auto residual_vector = Kokkos::View<double[num_elements * num_nodes * 6]>("residual_vector");

    Kokkos::parallel_for(
        num_elements * num_nodes,
        IntegrateResidualVector{
            node_state_indices, node_FE, node_FI, node_FG, node_FX, residual_vector}
    );

    expect_kokkos_view_1D_equal(residual_vector, {0., 0., 0., 0., 0., 0.});
}

TEST(IntegrateResidualVector, OneElementOneNode_Sum) {
    TestIntegrateResidualVector_1Element1Node_Sum();
}

void TestIntegrateResidualVector_OneElementTwoNodes() {
    constexpr auto num_elements = size_t{1U};
    constexpr auto num_nodes = size_t{2U};
    const auto node_state_indices = get_node_state_indices<num_elements, num_nodes>();

    const auto node_FE =
        get_node_FE<num_elements, num_nodes>({1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    const auto node_FI = Kokkos::View<double[num_elements * num_nodes][6]>("node_FI");
    const auto node_FG = Kokkos::View<double[num_elements * num_nodes][6]>("node_FG");
    const auto node_FX = Kokkos::View<double[num_elements * num_nodes][6]>("node_FX");

    auto residual_vector = Kokkos::View<double[num_elements * num_nodes * 6]>("residual_vector");

    Kokkos::parallel_for(
        num_elements * num_nodes,
        IntegrateResidualVector{
            node_state_indices, node_FE, node_FI, node_FG, node_FX, residual_vector}
    );

    expect_kokkos_view_1D_equal(
        residual_vector, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.}
    );
}

TEST(IntegrateResidualVector, OneElementTwoNodes) {
    TestIntegrateResidualVector_OneElementTwoNodes();
}

void TestIntegrateResidualVector_2Elements2Nodes() {
    constexpr auto num_elements = size_t{2U};
    constexpr auto num_nodes = size_t{2U};
    const auto node_state_indices = get_node_state_indices<num_elements, num_nodes>();

    const auto node_FE =
        get_node_FE<num_elements, num_nodes>({1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                                              9.,  10., 11., 12., 13., 14., 15., 16.,
                                              17., 18., 19., 20., 21., 22., 23., 24.});
    const auto node_FI = Kokkos::View<double[num_elements * num_nodes][6]>("node_FI");
    const auto node_FG = Kokkos::View<double[num_elements * num_nodes][6]>("node_FG");
    const auto node_FX = Kokkos::View<double[num_elements * num_nodes][6]>("node_FX");

    auto residual_vector = Kokkos::View<double[num_elements * num_nodes * 6]>("residual_vector");

    Kokkos::parallel_for(
        num_elements * num_nodes,
        IntegrateResidualVector{
            node_state_indices, node_FE, node_FI, node_FG, node_FX, residual_vector}
    );

    expect_kokkos_view_1D_equal(residual_vector, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                                                  9.,  10., 11., 12., 13., 14., 15., 16.,
                                                  17., 18., 19., 20., 21., 22., 23., 24.});
}

TEST(IntegrateResidualVector, TwoElementsTwoNodes) {
    TestIntegrateResidualVector_2Elements2Nodes();
}

}  // namespace openturbine::tests