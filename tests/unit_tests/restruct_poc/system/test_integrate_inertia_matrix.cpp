#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"
#include "test_integrate_matrix.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/integrate_inertia_matrix.hpp"
#include "src/restruct_poc/types.hpp"
#include "tests/unit_tests/restruct_poc/test_utilities.hpp"

namespace openturbine::tests {

inline void IntegrateInertiaMatrix_TestOneElementOneNodeOneQP_Muu() {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({3.});
    const auto shape_interp = get_shape_interp<number_of_nodes, number_of_qps>({5.});
    const auto qp_Muu =
        get_qp_Muu<number_of_qps>({0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003.,
                                   1004., 1005., 1006., 2001., 2002., 2003., 2004., 2005., 2006.,
                                   3001., 3002., 3003., 3004., 3005., 3006., 4001., 4002., 4003.,
                                   4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.});
    const auto qp_Guu =
        get_qp_Guu<number_of_qps>({1001., 1002., 1003., 1004., 1005., 1006., 2001., 2002., 2003.,
                                   2004., 2005., 2006., 4001., 4002., 4003., 4004., 4005., 4006.,
                                   5001., 5002., 5003., 5004., 5005., 5006., 6001., 6002., 6003.,
                                   6004., 6005., 6006., 7001., 7002., 7003., 7004., 7005., 7006.});

    const auto gbl_M = Kokkos::View<double[1][1][1][6][6]>("global_M");

    const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
    const auto integrator = IntegrateInertiaMatrixElement{
        0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, qp_Muu, qp_Guu, 1., 0., gbl_M};
    Kokkos::parallel_for(policy, integrator);

    constexpr auto exact_M_data =
        std::array{000150., 000300., 000450., 000600., 000750., 000900., 150150., 150300., 150450.,
                   150600., 150750., 150900., 300150., 300300., 300450., 300600., 300750., 300900.,
                   450150., 450300., 450450., 450600., 450750., 450900., 600150., 600300., 600450.,
                   600600., 600750., 600900., 750150., 750300., 750450., 750600., 750750., 750900.};
    const auto exact_M = Kokkos::View<const double[1][1][1][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

TEST(IntegrateInertiaMatrixTests, OneElementOneNodeOneQP_Muu) {
    IntegrateInertiaMatrix_TestOneElementOneNodeOneQP_Muu();
}

void IntegrateInertiaMatrix_TestOneElementOneNodeOneQP_Guu() {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({3.});
    const auto shape_interp = get_shape_interp<number_of_nodes, number_of_qps>({5.});
    const auto qp_Muu =
        get_qp_Muu<number_of_qps>({0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003.,
                                   1004., 1005., 1006., 2001., 2002., 2003., 2004., 2005., 2006.,
                                   3001., 3002., 3003., 3004., 3005., 3006., 4001., 4002., 4003.,
                                   4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.});
    const auto qp_Guu =
        get_qp_Guu<number_of_qps>({1001., 1002., 1003., 1004., 1005., 1006., 2001., 2002., 2003.,
                                   2004., 2005., 2006., 3001., 3002., 3003., 3004., 3005., 3006.,
                                   4001., 4002., 4003., 4004., 4005., 4006., 5001., 5002., 5003.,
                                   5004., 5005., 5006., 6001., 6002., 6003., 6004., 6005., 6006.});

    const auto gbl_M = Kokkos::View<double[1][1][1][6][6]>("global_M");

    const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
    const auto integrator = IntegrateInertiaMatrixElement{
        0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, qp_Muu, qp_Guu, 0., 1., gbl_M};
    Kokkos::parallel_for(policy, integrator);

    constexpr auto exact_M_data =
        std::array{150150., 150300., 150450., 150600., 150750., 150900., 300150., 300300., 300450.,
                   300600., 300750., 300900., 450150., 450300., 450450., 450600., 450750., 450900.,
                   600150., 600300., 600450., 600600., 600750., 600900., 750150., 750300., 750450.,
                   750600., 750750., 750900., 900150., 900300., 900450., 900600., 900750., 900900.};
    const auto exact_M = Kokkos::View<const double[1][1][1][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

TEST(IntegrateInertiaMatrixTests, OneElementOneNodeOneQP_Guu) {
    IntegrateInertiaMatrix_TestOneElementOneNodeOneQP_Guu();
}

void IntegrateInertiaMatrix_TestTwoElementsOneNodeOneQP() {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({1.});
    const auto shape_interp = get_shape_interp<number_of_nodes, number_of_qps>({1.});
    using QpMatrixView = Kokkos::View<double[number_of_qps][6][6]>;

    const auto qp_Guu = QpMatrixView("Guu");
    const auto gbl_M = Kokkos::View<double[2][1][1][6][6]>("global_M");

    {
        const auto qp_Muu = get_qp_Muu<number_of_qps>(
            {00001., 00002., 00003., 00004., 00005., 00006., 00101., 00102., 00103.,
             00104., 00105., 00106., 00201., 00202., 00203., 00204., 00205., 00206.,
             00301., 00302., 00303., 00304., 00305., 00306., 00401., 00402., 00403.,
             00404., 00405., 00406., 00501., 00502., 00503., 00504., 00505., 00506.}
        );
        const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
        const auto integrator = IntegrateInertiaMatrixElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, qp_Muu, qp_Guu, 1., 0., gbl_M};
        Kokkos::parallel_for(policy, integrator);
    }

    {
        const auto qp_Muu = get_qp_Muu<number_of_qps>(
            {00001., 00002., 00003., 00004., 00005., 00006., 10001., 10002., 10003.,
             10004., 10005., 10006., 20001., 20002., 20003., 20004., 20005., 20006.,
             30001., 30002., 30003., 30004., 30005., 30006., 40001., 40002., 40003.,
             40004., 40005., 40006., 50001., 50002., 50003., 50004., 50005., 50006.}
        );
        const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
        const auto integrator = IntegrateInertiaMatrixElement{
            1U, number_of_qps, qp_weights, qp_jacobian, shape_interp, qp_Muu, qp_Guu, 1., 0., gbl_M};
        Kokkos::parallel_for(policy, integrator);
    }

    constexpr auto exact_M_data =
        std::array{00001., 00002., 00003., 00004., 00005., 00006., 00101., 00102., 00103.,
                   00104., 00105., 00106., 00201., 00202., 00203., 00204., 00205., 00206.,
                   00301., 00302., 00303., 00304., 00305., 00306., 00401., 00402., 00403.,
                   00404., 00405., 00406., 00501., 00502., 00503., 00504., 00505., 00506.,

                   00001., 00002., 00003., 00004., 00005., 00006., 10001., 10002., 10003.,
                   10004., 10005., 10006., 20001., 20002., 20003., 20004., 20005., 20006.,
                   30001., 30002., 30003., 30004., 30005., 30006., 40001., 40002., 40003.,
                   40004., 40005., 40006., 50001., 50002., 50003., 50004., 50005., 50006.};
    const auto exact_M = Kokkos::View<const double[2][1][1][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

TEST(IntegrateInertiaMatrixTests, TwoElementsOneNodeOneQP) {
    IntegrateInertiaMatrix_TestTwoElementsOneNodeOneQP();
}

void IntegrateInertiaMatrix_TestOneElementTwoNodesOneQP() {
    constexpr auto number_of_nodes = size_t{2U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({1.});
    const auto shape_interp = get_shape_interp<number_of_nodes, number_of_qps>({1., 2.});
    using QpMatrixView = Kokkos::View<double[number_of_qps][6][6]>;
    const auto qp_Muu =
        get_qp_Muu<number_of_qps>({0001., 0002., 0003., 0004., 0005., 0006., 0101., 0102., 0103.,
                                   0104., 0105., 0106., 0201., 0202., 0203., 0204., 0205., 0206.,
                                   0301., 0302., 0303., 0304., 0305., 0306., 0401., 0402., 0403.,
                                   0404., 0405., 0406., 0501., 0502., 0503., 0504., 0505., 0506.});
    const auto qp_Guu = QpMatrixView("Guu");

    const auto gbl_M = Kokkos::View<double[1][2][2][6][6]>("global_M");

    const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
    const auto integrator = IntegrateInertiaMatrixElement{
        0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, qp_Muu, qp_Guu, 1., 0., gbl_M};
    Kokkos::parallel_for(policy, integrator);

    constexpr auto exact_M_data = std::array<double, 144>{1., 2., 3., 4., 5., 6.,
                                                        101., 102., 103., 104., 105., 106.,
                                                        201., 202., 203., 204., 205., 206.,
                                                        301., 302., 303., 304., 305., 306.,
                                                        401., 402., 403., 404., 405., 406.,
                                                        501., 502., 503., 504., 505., 506.,
                                                          2., 4., 6., 8., 10., 12.,
                                                        202., 204., 206., 208., 210., 212.,
                                                        402., 404., 406., 408., 410., 412.,
                                                        602., 604., 606., 608., 610., 612.,
                                                        802., 804., 806., 808., 810., 812.,
                                                        1002., 1004., 1006., 1008., 1010., 1012.,
                                                          2., 4., 6., 8., 10., 12.,
                                                        202., 204., 206., 208., 210., 212.,
                                                        402., 404., 406., 408., 410., 412.,
                                                        602., 604., 606., 608., 610., 612.,
                                                        802., 804., 806., 808., 810., 812.,
                                                        1002., 1004., 1006., 1008., 1010., 1012.,
                                                          4., 8., 12., 16., 20., 24.,
                                                        404., 408., 412., 416., 420., 424.,
                                                        804., 808., 812., 816., 820., 824.,
                                                        1204., 1208., 1212., 1216., 1220., 1224.,
                                                        1604., 1608., 1612., 1616., 1620., 1624.,
                                                        2004., 2008., 2012., 2016., 2020., 2024.};

    const auto exact_M =
        Kokkos::View<const double[1][2][2][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

TEST(IntegrateInertiaMatrixTests, OneElementTwoNodesOneQP) {
    IntegrateInertiaMatrix_TestOneElementTwoNodesOneQP();
}

void IntegrateInertiaMatrix_TestOneElementOneNodeTwoQPs() {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{2U};

    const auto qp_weights = get_qp_weights<number_of_qps>({9., 1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({1., 4.});
    const auto shape_interp = get_shape_interp<number_of_nodes, number_of_qps>({1. / 3., 1. / 2.});
    using QpMatrixView = Kokkos::View<double[number_of_qps][6][6]>;
    const auto qp_Muu = get_qp_Muu<number_of_qps>(
        {00001., 00002., 00003., 00004., 00005., 00006., 00011., 00012., 00013.,
         00014., 00015., 00016., 00021., 00022., 00023., 00024., 00025., 00026.,
         00031., 00032., 00033., 00034., 00035., 00036., 00041., 00042., 00043.,
         00044., 00045., 00046., 00051., 00052., 00053., 00054., 00055., 00056.,

         01000., 02000., 03000., 04000., 05000., 06000., 11000., 12000., 13000.,
         14000., 15000., 16000., 21000., 22000., 23000., 24000., 25000., 26000.,
         31000., 32000., 33000., 34000., 35000., 36000., 41000., 42000., 43000.,
         44000., 45000., 46000., 51000., 52000., 53000., 54000., 55000., 56000.}
    );
    const auto qp_Guu = QpMatrixView("Guu");

    const auto gbl_M = Kokkos::View<double[1][1][1][6][6]>("global_M");

    const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
    const auto integrator = IntegrateInertiaMatrixElement{
        0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, qp_Muu, qp_Guu, 1., 0., gbl_M};
    Kokkos::parallel_for(policy, integrator);

    constexpr auto exact_M_data =
        std::array{01001., 02002., 03003., 04004., 05005., 06006., 11011., 12012., 13013.,
                   14014., 15015., 16016., 21021., 22022., 23023., 24024., 25025., 26026.,
                   31031., 32032., 33033., 34034., 35035., 36036., 41041., 42042., 43043.,
                   44044., 45045., 46046., 51051., 52052., 53053., 54054., 55055., 56056.};
    const auto exact_M = Kokkos::View<const double[1][1][1][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

TEST(IntegrateInertiaMatrixTests, OneElementOneNodeTwoQPs) {
    IntegrateInertiaMatrix_TestOneElementOneNodeTwoQPs();
}

void IntegrateInertiaMatrix_TestOneElementOneNodeOneQP_WithMultiplicationFactor() {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1};

    const auto qp_weights = get_qp_weights<number_of_qps>({1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({1.});
    const auto shape_interp = get_shape_interp<number_of_nodes, number_of_qps>({1.});
    using QpMatrixView = Kokkos::View<double[number_of_qps][6][6]>;
    const auto qp_Muu =
        get_qp_Muu<number_of_qps>({0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003.,
                                   1004., 1005., 1006., 2001., 2002., 2003., 2004., 2005., 2006.,
                                   3001., 3002., 3003., 3004., 3005., 3006., 4001., 4002., 4003.,
                                   4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.});
    const auto qp_Guu = QpMatrixView("Guu");
    const auto multiplication_factor = 5.;

    const auto gbl_M = Kokkos::View<double[1][1][1][6][6]>("global_M");

    const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
    const auto integrator = IntegrateInertiaMatrixElement{
        0U,     number_of_qps,         qp_weights, qp_jacobian, shape_interp, qp_Muu,
        qp_Guu, multiplication_factor, 0.,         gbl_M};
    Kokkos::parallel_for(policy, integrator);

    constexpr auto exact_M_data =
        std::array{00005., 00010., 00015., 00020., 00025., 00030., 05005., 05010., 05015.,
                   05020., 05025., 05030., 10005., 10010., 10015., 10020., 10025., 10030.,
                   15005., 15010., 15015., 15020., 15025., 15030., 20005., 20010., 20015.,
                   20020., 20025., 20030., 25005., 25010., 25015., 25020., 25025., 25030.};
    const auto exact_M = Kokkos::View<const double[1][1][1][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

TEST(IntegrateInertiaMatrixTests, OneElementOneNodeOneQP_WithMultiplicationFactor_TeamPolicy) {
    IntegrateInertiaMatrix_TestOneElementOneNodeOneQP_WithMultiplicationFactor();
}

}  // namespace openturbine::tests
