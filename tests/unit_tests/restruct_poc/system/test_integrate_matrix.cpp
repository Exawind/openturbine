#include "test_integrate_matrix.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/integrate_matrix.hpp"
#include "src/restruct_poc/types.hpp"
#include "tests/unit_tests/restruct_poc/test_utilities.hpp"

namespace openturbine::restruct_poc::tests {

template <typename Policy>
void TestOneElementOneNodeOneQP(Policy policy) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 1;

    const auto element_indices =
        get_element_indices<number_of_elements, number_of_nodes, number_of_qps>();
    const auto node_state_indices = get_node_state_indices<number_of_elements, number_of_nodes>();
    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({3.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({5.});
    const auto qp_M = get_qp_M<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003., 1004., 1005., 1006.,
         2001., 2002., 2003., 2004., 2005., 2006., 3001., 3002., 3003., 3004., 3005., 3006.,
         4001., 4002., 4003., 4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.}
    );

    auto gbl_M = Kokkos::View<double[6][6]>("global_M");

    Kokkos::parallel_for(
        policy,
        IntegrateMatrix{
            element_indices, node_state_indices, qp_weights, qp_jacobian, shape_interp, qp_M, gbl_M}
    );

    expect_kokkos_view_2D_equal(
        gbl_M, {{000150., 000300., 000450., 000600., 000750., 000900.},
                {150150., 150300., 150450., 150600., 150750., 150900.},
                {300150., 300300., 300450., 300600., 300750., 300900.},
                {450150., 450300., 450450., 450600., 450750., 450900.},
                {600150., 600300., 600450., 600600., 600750., 600900.},
                {750150., 750300., 750450., 750600., 750750., 750900.}}
    );
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_1D) {
    TestOneElementOneNodeOneQP(Kokkos::RangePolicy(0, 1));
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_3D) {
    TestOneElementOneNodeOneQP(Kokkos::MDRangePolicy{{0, 0, 0}, {1, 1, 1}});
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_4D) {
    TestOneElementOneNodeOneQP(Kokkos::MDRangePolicy{{0, 0, 0, 0}, {1, 1, 1, 1}});
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_TeamPolicy) {
    TestOneElementOneNodeOneQP(Kokkos::TeamPolicy<>(1, Kokkos::AUTO()));
}

template <typename Policy>
void TestTwoElementsOneNodeOneQP(Policy policy) {
    constexpr auto number_of_elements = 2;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 1;

    const auto element_indices =
        get_element_indices<number_of_elements, number_of_nodes, number_of_qps>();
    const auto node_state_indices = get_node_state_indices<number_of_elements, number_of_nodes>();
    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({1., 1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({1., 1.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({1., 0., 1., 0.});
    const auto qp_M = get_qp_M<number_of_elements, number_of_qps>(
        {00001., 00002., 00003., 00004., 00005., 00006., 00101., 00102., 00103.,
         00104., 00105., 00106., 00201., 00202., 00203., 00204., 00205., 00206.,
         00301., 00302., 00303., 00304., 00305., 00306., 00401., 00402., 00403.,
         00404., 00405., 00406., 00501., 00502., 00503., 00504., 00505., 00506.,

         00001., 00002., 00003., 00004., 00005., 00006., 10001., 10002., 10003.,
         10004., 10005., 10006., 20001., 20002., 20003., 20004., 20005., 20006.,
         30001., 30002., 30003., 30004., 30005., 30006., 40001., 40002., 40003.,
         40004., 40005., 40006., 50001., 50002., 50003., 50004., 50005., 50006.}
    );

    auto gbl_M = Kokkos::View<double[12][12]>("global_M");

    Kokkos::parallel_for(
        policy,
        IntegrateMatrix{
            element_indices, node_state_indices, qp_weights, qp_jacobian, shape_interp, qp_M, gbl_M}
    );

    expect_kokkos_view_2D_equal(
        gbl_M, {{00001., 00002., 00003., 00004., 00005., 00006., 0., 0., 0., 0., 0., 0.},
                {00101., 00102., 00103., 00104., 00105., 00106., 0., 0., 0., 0., 0., 0.},
                {00201., 00202., 00203., 00204., 00205., 00206., 0., 0., 0., 0., 0., 0.},
                {00301., 00302., 00303., 00304., 00305., 00306., 0., 0., 0., 0., 0., 0.},
                {00401., 00402., 00403., 00404., 00405., 00406., 0., 0., 0., 0., 0., 0.},
                {00501., 00502., 00503., 00504., 00505., 00506., 0., 0., 0., 0., 0., 0.},

                {0., 0., 0., 0., 0., 0., 00001., 00002., 00003., 00004., 00005., 00006.},
                {0., 0., 0., 0., 0., 0., 10001., 10002., 10003., 10004., 10005., 10006.},
                {0., 0., 0., 0., 0., 0., 20001., 20002., 20003., 20004., 20005., 20006.},
                {0., 0., 0., 0., 0., 0., 30001., 30002., 30003., 30004., 30005., 30006.},
                {0., 0., 0., 0., 0., 0., 40001., 40002., 40003., 40004., 40005., 40006.},
                {0., 0., 0., 0., 0., 0., 50001., 50002., 50003., 50004., 50005., 50006.}}
    );
}

TEST(IntegrateMatrixTests, TwoElementsOneNodeOneQP_1D) {
    TestTwoElementsOneNodeOneQP(Kokkos::RangePolicy(0, 2));
}

TEST(IntegrateMatrixTests, TwoElementsOneNodeOneQP_3D) {
    TestTwoElementsOneNodeOneQP(Kokkos::MDRangePolicy{{0, 0, 0}, {2, 1, 1}});
}

TEST(IntegrateMatrixTests, TwoElementsOneNodeOneQP_4D) {
    TestTwoElementsOneNodeOneQP(Kokkos::MDRangePolicy{{0, 0, 0, 0}, {2, 1, 1, 1}});
}

TEST(IntegrateMatrixTests, TwoElementsOneNodeOneQP_TeamPolicy) {
    TestTwoElementsOneNodeOneQP(Kokkos::TeamPolicy<>(2, Kokkos::AUTO()));
}

template <typename Policy>
void TestOneElementTwoNodesOneQP(Policy policy) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 2;
    constexpr auto number_of_qps = 1;

    const auto element_indices =
        get_element_indices<number_of_elements, number_of_nodes, number_of_qps>();
    const auto node_state_indices = get_node_state_indices<number_of_elements, number_of_nodes>();
    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({1.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({1., 2.});
    const auto qp_M = get_qp_M<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 0101., 0102., 0103., 0104., 0105., 0106.,
         0201., 0202., 0203., 0204., 0205., 0206., 0301., 0302., 0303., 0304., 0305., 0306.,
         0401., 0402., 0403., 0404., 0405., 0406., 0501., 0502., 0503., 0504., 0505., 0506.}
    );

    auto gbl_M = Kokkos::View<double[12][12]>("global_M");

    Kokkos::parallel_for(
        policy,
        IntegrateMatrix{
            element_indices, node_state_indices, qp_weights, qp_jacobian, shape_interp, qp_M, gbl_M}
    );

    expect_kokkos_view_2D_equal(
        gbl_M, {{0001., 0002., 0003., 0004., 0005., 0006., 0002., 0004., 0006., 0008., 0010., 0012.},
                {0101., 0102., 0103., 0104., 0105., 0106., 0202., 0204., 0206., 0208., 0210., 0212.},
                {0201., 0202., 0203., 0204., 0205., 0206., 0402., 0404., 0406., 0408., 0410., 0412.},
                {0301., 0302., 0303., 0304., 0305., 0306., 0602., 0604., 0606., 0608., 0610., 0612.},
                {0401., 0402., 0403., 0404., 0405., 0406., 0802., 0804., 0806., 0808., 0810., 0812.},
                {0501., 0502., 0503., 0504., 0505., 0506., 1002., 1004., 1006., 1008., 1010., 1012.},

                {0002., 0004., 0006., 0008., 0010., 0012., 0004., 0008., 0012., 0016., 0020., 0024.},
                {0202., 0204., 0206., 0208., 0210., 0212., 0404., 0408., 0412., 0416., 0420., 0424.},
                {0402., 0404., 0406., 0408., 0410., 0412., 0804., 0808., 0812., 0816., 0820., 0824.},
                {0602., 0604., 0606., 0608., 0610., 0612., 1204., 1208., 1212., 1216., 1220., 1224.},
                {0802., 0804., 0806., 0808., 0810., 0812., 1604., 1608., 1612., 1616., 1620., 1624.},
                {1002., 1004., 1006., 1008., 1010., 1012., 2004., 2008., 2012., 2016., 2020., 2024.}}
    );
}

TEST(IntegrateMatrixTests, OneElementTwoNodesOneQP_1D) {
    TestOneElementTwoNodesOneQP(Kokkos::RangePolicy(0, 1));
}

TEST(IntegrateMatrixTests, OneElementTwoNodesOneQP_3D) {
    TestOneElementTwoNodesOneQP(Kokkos::MDRangePolicy{{0, 0, 0}, {1, 2, 2}});
}

TEST(IntegrateMatrixTests, OneElementTwoNodesOneQP_4D) {
    TestOneElementTwoNodesOneQP(Kokkos::MDRangePolicy{{0, 0, 0, 0}, {1, 2, 2, 1}});
}

TEST(IntegrateMatrixTests, OneElementTwoNodesOneQP_TeamPolicy) {
    TestOneElementTwoNodesOneQP(Kokkos::TeamPolicy<>(1, Kokkos::AUTO()));
}

template <typename Policy>
void TestOneElementOneNodeTwoQPs(Policy policy) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 2;

    const auto element_indices =
        get_element_indices<number_of_elements, number_of_nodes, number_of_qps>();
    const auto node_state_indices = get_node_state_indices<number_of_elements, number_of_nodes>();
    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({9., 1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({1., 4.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({1. / 3., 1. / 2.});
    const auto qp_M = get_qp_M<number_of_elements, number_of_qps>(
        {00001., 00002., 00003., 00004., 00005., 00006., 00011., 00012., 00013.,
         00014., 00015., 00016., 00021., 00022., 00023., 00024., 00025., 00026.,
         00031., 00032., 00033., 00034., 00035., 00036., 00041., 00042., 00043.,
         00044., 00045., 00046., 00051., 00052., 00053., 00054., 00055., 00056.,

         01000., 02000., 03000., 04000., 05000., 06000., 11000., 12000., 13000.,
         14000., 15000., 16000., 21000., 22000., 23000., 24000., 25000., 26000.,
         31000., 32000., 33000., 34000., 35000., 36000., 41000., 42000., 43000.,
         44000., 45000., 46000., 51000., 52000., 53000., 54000., 55000., 56000.}
    );

    auto gbl_M = Kokkos::View<double[6][6]>("global_M");

    Kokkos::parallel_for(
        policy,
        IntegrateMatrix{
            element_indices, node_state_indices, qp_weights, qp_jacobian, shape_interp, qp_M, gbl_M}
    );

    expect_kokkos_view_2D_equal(
        gbl_M, {{01001., 02002., 03003., 04004., 05005., 06006.},
                {11011., 12012., 13013., 14014., 15015., 16016.},
                {21021., 22022., 23023., 24024., 25025., 26026.},
                {31031., 32032., 33033., 34034., 35035., 36036.},
                {41041., 42042., 43043., 44044., 45045., 46046.},
                {51051., 52052., 53053., 54054., 55055., 56056.}}
    );
}

TEST(IntegrateMatrixTests, OneElementOneNodeTwoQPs_1D) {
    TestOneElementOneNodeTwoQPs(Kokkos::RangePolicy(0, 1));
}

TEST(IntegrateMatrixTests, OneElementOneNodeTwoQPs_3D) {
    TestOneElementOneNodeTwoQPs(Kokkos::MDRangePolicy{{0, 0, 0}, {1, 1, 1}});
}

TEST(IntegrateMatrixTests, OneElementOneNodeTwoQPs_4D) {
    TestOneElementOneNodeTwoQPs(Kokkos::MDRangePolicy{{0, 0, 0, 0}, {1, 1, 1, 2}});
}

TEST(IntegrateMatrixTests, OneElementOneNodeTwoQPs_TeamPolicy) {
    TestOneElementOneNodeTwoQPs(Kokkos::TeamPolicy<>(1, Kokkos::AUTO()));
}

template <typename Policy>
void TestOneElementOneNodeOneQP_WithMultiplicationFactor(Policy policy) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 1;

    const auto element_indices =
        get_element_indices<number_of_elements, number_of_nodes, number_of_qps>();
    const auto node_state_indices = get_node_state_indices<number_of_elements, number_of_nodes>();
    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({1.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({1.});
    const auto qp_M = get_qp_M<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003., 1004., 1005., 1006.,
         2001., 2002., 2003., 2004., 2005., 2006., 3001., 3002., 3003., 3004., 3005., 3006.,
         4001., 4002., 4003., 4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.}
    );
    const auto multiplication_factor = 5.;

    auto gbl_M = Kokkos::View<double[6][6]>("global_M");

    Kokkos::parallel_for(
        policy,
        IntegrateMatrix{
            element_indices, node_state_indices, qp_weights, qp_jacobian, shape_interp, qp_M, gbl_M,
            multiplication_factor}
    );

    expect_kokkos_view_2D_equal(
        gbl_M, {{00005., 00010., 00015., 00020., 00025., 00030.},
                {05005., 05010., 05015., 05020., 05025., 05030.},
                {10005., 10010., 10015., 10020., 10025., 10030.},
                {15005., 15010., 15015., 15020., 15025., 15030.},
                {20005., 20010., 20015., 20020., 20025., 20030.},
                {25005., 25010., 25015., 25020., 25025., 25030.}}
    );
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_WithMultiplicationFactor_1D) {
    TestOneElementOneNodeOneQP_WithMultiplicationFactor(Kokkos::RangePolicy(0, 1));
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_WithMultiplicationFactor_3D) {
    TestOneElementOneNodeOneQP_WithMultiplicationFactor(Kokkos::MDRangePolicy{{0, 0, 0}, {1, 1, 1}});
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_WithMultiplicationFactor_4D) {
    TestOneElementOneNodeOneQP_WithMultiplicationFactor(Kokkos::MDRangePolicy{
        {0, 0, 0, 0}, {1, 1, 1, 1}});
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_WithMultiplicationFactor_TeamPolicy) {
    TestOneElementOneNodeOneQP_WithMultiplicationFactor(Kokkos::TeamPolicy<>(1, Kokkos::AUTO()));
}

}  // namespace openturbine::restruct_poc::tests
