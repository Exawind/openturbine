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
    const auto element_indices = std::invoke(
        [](int n_elem, int n_nodes, int n_qps) {
            auto elem_indices = Kokkos::View<Beams::ElemIndices*>("elem_indices", n_elem);
            auto host_elem_indices = Kokkos::create_mirror(elem_indices);
            host_elem_indices(0) = Beams::ElemIndices(n_nodes, n_qps, 0, 0);
            Kokkos::deep_copy(elem_indices, host_elem_indices);
            return elem_indices;
        },
        number_of_elements, number_of_nodes, number_of_qps
    );

    const auto node_state_indices = std::invoke(
        [](int n_elem) {
            auto indices = Kokkos::View<int*>("node_state_indices", n_elem);
            auto host_indices = Kokkos::create_mirror(indices);
            host_indices(0) = 0;
            Kokkos::deep_copy(indices, host_indices);
            return indices;
        },
        number_of_elements
    );

    const auto qp_weights = std::invoke(
        [](int n_qps) {
            auto weights = Kokkos::View<double*>("weights", n_qps);
            auto host_weights = Kokkos::create_mirror(weights);
            host_weights(0) = 2.;
            Kokkos::deep_copy(weights, host_weights);
            return weights;
        },
        number_of_qps
    );

    const auto qp_jacobian = std::invoke(
        [](int n_qps) {
            auto jacobian = Kokkos::View<double*>("jacobian", n_qps);
            auto host_jacobian = Kokkos::create_mirror(jacobian);
            host_jacobian(0) = 3.;
            Kokkos::deep_copy(jacobian, host_jacobian);
            return jacobian;
        },
        number_of_qps
    );

    const auto shape_interp = std::invoke(
        [](int n_nodes, int n_qps) {
            auto shape = Kokkos::View<double**>("shape_interp", n_nodes, n_qps);
            auto host_shape = Kokkos::create_mirror(shape);
            host_shape(0, 0) = 5.;
            Kokkos::deep_copy(shape, host_shape);
            return shape;
        },
        number_of_nodes, number_of_qps
    );

    const auto qp_M = std::invoke(
        [](int n_qps) {
            auto M = Kokkos::View<double* [6][6]>("M", n_qps);
            auto host_M = Kokkos::create_mirror(M);
            auto M_data = std::array{0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003.,
                                     1004., 1005., 1006., 2001., 2002., 2003., 2004., 2005., 2006.,
                                     3001., 3002., 3003., 3004., 3005., 3006., 4001., 4002., 4003.,
                                     4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.};
            using data_view_type = Kokkos::View<
                double[1][6][6], Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            auto M_data_view = data_view_type(M_data.data());
            Kokkos::deep_copy(host_M, M_data_view);
            Kokkos::deep_copy(M, host_M);
            return M;
        },
        number_of_qps
    );
    auto gbl_M = Kokkos::View<double**>("global_M", 6, 6);

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

template <typename Policy>
void TestTwoElementsOneNodeOneQP(Policy policy) {
    constexpr auto number_of_elements = 2;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 1;
    const auto element_indices = std::invoke(
        [](int n_elem, int n_nodes, int n_qps) {
            auto elem_indices = Kokkos::View<Beams::ElemIndices*>("elem_indices", n_elem);
            auto host_elem_indices = Kokkos::create_mirror(elem_indices);
            host_elem_indices(0) = Beams::ElemIndices(n_nodes, n_qps, 0, 0);
            host_elem_indices(1) = Beams::ElemIndices(n_nodes, n_qps, 1, 1);
            Kokkos::deep_copy(elem_indices, host_elem_indices);
            return elem_indices;
        },
        number_of_elements, number_of_nodes, number_of_qps
    );

    const auto node_state_indices = std::invoke(
        [](int n_elem) {
            auto indices = Kokkos::View<int*>("node_state_indices", n_elem);
            auto host_indices = Kokkos::create_mirror(indices);
            host_indices(0) = 0;
            host_indices(1) = 1;
            Kokkos::deep_copy(indices, host_indices);
            return indices;
        },
        number_of_elements
    );

    const auto qp_weights = std::invoke(
        [](int n_elem, int n_qps) {
            auto weights = Kokkos::View<double*>("weights", n_elem * n_qps);
            auto host_weights = Kokkos::create_mirror(weights);
            host_weights(0) = 1.;
            host_weights(1) = 1.;
            Kokkos::deep_copy(weights, host_weights);
            return weights;
        },
        number_of_elements, number_of_qps
    );

    const auto qp_jacobian = std::invoke(
        [](int n_elem, int n_qps) {
            auto jacobian = Kokkos::View<double*>("jacobian", n_elem * n_qps);
            auto host_jacobian = Kokkos::create_mirror(jacobian);
            host_jacobian(0) = 1.;
            host_jacobian(1) = 1.;
            Kokkos::deep_copy(jacobian, host_jacobian);
            return jacobian;
        },
        number_of_elements, number_of_qps
    );

    const auto shape_interp = std::invoke(
        [](int n_elem, int n_nodes, int n_qps) {
            auto shape = Kokkos::View<double**>("shape_interp", n_elem * n_nodes, n_elem * n_qps);
            auto host_shape = Kokkos::create_mirror(shape);
            host_shape(0, 0) = 1.;
            host_shape(1, 0) = 1.;
            Kokkos::deep_copy(shape, host_shape);
            return shape;
        },
        number_of_elements, number_of_nodes, number_of_qps
    );

    const auto qp_M = std::invoke(
        [](int n_elem, int n_qps) {
            auto M = Kokkos::View<double* [6][6]>("M", n_elem * n_qps);
            auto host_M = Kokkos::create_mirror(M);
            auto M_data =
                std::array{00001., 00002., 00003., 00004., 00005., 00006., 00101., 00102., 00103.,
                           00104., 00105., 00106., 00201., 00202., 00203., 00204., 00205., 00206.,
                           00301., 00302., 00303., 00304., 00305., 00306., 00401., 00402., 00403.,
                           00404., 00405., 00406., 00501., 00502., 00503., 00504., 00505., 00506.,

                           00001., 00002., 00003., 00004., 00005., 00006., 10001., 10002., 10003.,
                           10004., 10005., 10006., 20001., 20002., 20003., 20004., 20005., 20006.,
                           30001., 30002., 30003., 30004., 30005., 30006., 40001., 40002., 40003.,
                           40004., 40005., 40006., 50001., 50002., 50003., 50004., 50005., 50006.};
            using data_view_type = Kokkos::View<
                double[2][6][6], Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            auto M_data_view = data_view_type(M_data.data());
            Kokkos::deep_copy(host_M, M_data_view);
            Kokkos::deep_copy(M, host_M);
            return M;
        },
        number_of_elements, number_of_qps
    );
    auto gbl_M = Kokkos::View<double**>("global_M", 12, 12);

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

template <typename Policy>
void TestOneElementTwoNodesOneQP(Policy policy) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 2;
    constexpr auto number_of_qps = 1;
    const auto element_indices = std::invoke(
        [](int n_elem, int n_nodes, int n_qps) {
            auto elem_indices = Kokkos::View<Beams::ElemIndices*>("elem_indices", n_elem);
            auto host_elem_indices = Kokkos::create_mirror(elem_indices);
            host_elem_indices(0) = Beams::ElemIndices(n_nodes, n_qps, 0, 0);
            Kokkos::deep_copy(elem_indices, host_elem_indices);
            return elem_indices;
        },
        number_of_elements, number_of_nodes, number_of_qps
    );

    const auto node_state_indices = std::invoke(
        [](int n_elem, int n_nodes) {
            auto indices = Kokkos::View<int*>("node_state_indices", n_elem * n_nodes);
            auto host_indices = Kokkos::create_mirror(indices);
            host_indices(0) = 0;
            host_indices(1) = 1;
            Kokkos::deep_copy(indices, host_indices);
            return indices;
        },
        number_of_elements, number_of_nodes
    );

    const auto qp_weights = std::invoke(
        [](int n_elem, int n_qps) {
            auto weights = Kokkos::View<double*>("weights", n_elem * n_qps);
            auto host_weights = Kokkos::create_mirror(weights);
            host_weights(0) = 1.;
            Kokkos::deep_copy(weights, host_weights);
            return weights;
        },
        number_of_elements, number_of_qps
    );

    const auto qp_jacobian = std::invoke(
        [](int n_elem, int n_qps) {
            auto jacobian = Kokkos::View<double*>("jacobian", n_elem * n_qps);
            auto host_jacobian = Kokkos::create_mirror(jacobian);
            host_jacobian(0) = 1.;
            Kokkos::deep_copy(jacobian, host_jacobian);
            return jacobian;
        },
        number_of_elements, number_of_qps
    );

    const auto shape_interp = std::invoke(
        [](int n_elem, int n_nodes, int n_qps) {
            auto shape = Kokkos::View<double**>("shape_interp", n_elem * n_nodes, n_elem * n_qps);
            auto host_shape = Kokkos::create_mirror(shape);
            host_shape(0, 0) = 1.;
            host_shape(1, 0) = 2.;
            Kokkos::deep_copy(shape, host_shape);
            return shape;
        },
        number_of_elements, number_of_nodes, number_of_qps
    );

    const auto qp_M = std::invoke(
        [](int n_elem, int n_qps) {
            auto M = Kokkos::View<double* [6][6]>("M", n_elem * n_qps);
            auto host_M = Kokkos::create_mirror(M);
            auto M_data = std::array{0001., 0002., 0003., 0004., 0005., 0006., 0101., 0102., 0103.,
                                     0104., 0105., 0106., 0201., 0202., 0203., 0204., 0205., 0206.,
                                     0301., 0302., 0303., 0304., 0305., 0306., 0401., 0402., 0403.,
                                     0404., 0405., 0406., 0501., 0502., 0503., 0504., 0505., 0506.};
            using data_view_type = Kokkos::View<
                double[1][6][6], Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            auto M_data_view = data_view_type(M_data.data());
            Kokkos::deep_copy(host_M, M_data_view);
            Kokkos::deep_copy(M, host_M);
            return M;
        },
        number_of_elements, number_of_qps
    );
    auto gbl_M = Kokkos::View<double**>("global_M", 12, 12);

    Kokkos::parallel_for(
        policy,
        IntegrateMatrix{
            element_indices, node_state_indices, qp_weights, qp_jacobian, shape_interp, qp_M, gbl_M}
    );

    auto dbg = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(dbg, gbl_M);

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

template <typename Policy>
void TestOneElementOneNodeTwoQPs(Policy policy) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 2;
    const auto element_indices = std::invoke(
        [](int n_elem, int n_nodes, int n_qps) {
            auto elem_indices = Kokkos::View<Beams::ElemIndices*>("elem_indices", n_elem);
            auto host_elem_indices = Kokkos::create_mirror(elem_indices);
            host_elem_indices(0) = Beams::ElemIndices(n_nodes, n_qps, 0, 0);
            Kokkos::deep_copy(elem_indices, host_elem_indices);
            return elem_indices;
        },
        number_of_elements, number_of_nodes, number_of_qps
    );

    const auto node_state_indices = std::invoke(
        [](int n_elem) {
            auto indices = Kokkos::View<int*>("node_state_indices", n_elem);
            auto host_indices = Kokkos::create_mirror(indices);
            host_indices(0) = 0;
            Kokkos::deep_copy(indices, host_indices);
            return indices;
        },
        number_of_elements
    );

    const auto qp_weights = std::invoke(
        [](int n_elem, int n_qps) {
            auto weights = Kokkos::View<double*>("weights", n_elem * n_qps);
            auto host_weights = Kokkos::create_mirror(weights);
            host_weights(0) = 9.;
            host_weights(1) = 1.;
            Kokkos::deep_copy(weights, host_weights);
            return weights;
        },
        number_of_elements, number_of_qps
    );

    const auto qp_jacobian = std::invoke(
        [](int n_elem, int n_qps) {
            auto jacobian = Kokkos::View<double*>("jacobian", n_elem * n_qps);
            auto host_jacobian = Kokkos::create_mirror(jacobian);
            host_jacobian(0) = 1.;
            host_jacobian(1) = 4.;
            Kokkos::deep_copy(jacobian, host_jacobian);
            return jacobian;
        },
        number_of_elements, number_of_qps
    );

    const auto shape_interp = std::invoke(
        [](int n_elem, int n_nodes, int n_qps) {
            auto shape = Kokkos::View<double**>("shape_interp", n_elem * n_nodes, n_elem * n_qps);
            auto host_shape = Kokkos::create_mirror(shape);
            host_shape(0, 0) = 1. / 3.;
            host_shape(0, 1) = 1. / 2.;
            Kokkos::deep_copy(shape, host_shape);
            return shape;
        },
        number_of_elements, number_of_nodes, number_of_qps
    );

    const auto qp_M = std::invoke(
        [](int n_elem, int n_qps) {
            auto M = Kokkos::View<double* [6][6]>("M", n_elem * n_qps);
            auto host_M = Kokkos::create_mirror(M);
            auto M_data =
                std::array{00001., 00002., 00003., 00004., 00005., 00006., 00011., 00012., 00013.,
                           00014., 00015., 00016., 00021., 00022., 00023., 00024., 00025., 00026.,
                           00031., 00032., 00033., 00034., 00035., 00036., 00041., 00042., 00043.,
                           00044., 00045., 00046., 00051., 00052., 00053., 00054., 00055., 00056.,

                           01000., 02000., 03000., 04000., 05000., 06000., 11000., 12000., 13000.,
                           14000., 15000., 16000., 21000., 22000., 23000., 24000., 25000., 26000.,
                           31000., 32000., 33000., 34000., 35000., 36000., 41000., 42000., 43000.,
                           44000., 45000., 46000., 51000., 52000., 53000., 54000., 55000., 56000.};
            using data_view_type = Kokkos::View<
                double[2][6][6], Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            auto M_data_view = data_view_type(M_data.data());
            Kokkos::deep_copy(host_M, M_data_view);
            Kokkos::deep_copy(M, host_M);
            return M;
        },
        number_of_elements, number_of_qps
    );
    auto gbl_M = Kokkos::View<double**>("global_M", 6, 6);

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

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_OneIndex) {
    constexpr auto number_of_elements = 1;

    TestOneElementOneNodeOneQP(Kokkos::RangePolicy(0, number_of_elements));
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_TeamPolicy) {
    constexpr auto number_of_elements = 1;

    TestOneElementOneNodeOneQP(Kokkos::TeamPolicy<>(number_of_elements, Kokkos::AUTO()));
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_ThreeIndices) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;

    TestOneElementOneNodeOneQP(Kokkos::MDRangePolicy{
        {0, 0, 0}, {number_of_elements, number_of_nodes, number_of_nodes}});
}

TEST(IntegrateMatrixTests, OneElementOneNodeOneQP_FourIndices) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 1;

    TestOneElementOneNodeOneQP(Kokkos::MDRangePolicy{
        {0, 0, 0, 0}, {number_of_elements, number_of_nodes, number_of_nodes, number_of_qps}});
}

TEST(IntegrateMatrixTests, TestTwoElementsOneNodeOneQP_OneIndex) {
    constexpr auto number_of_elements = 2;

    TestTwoElementsOneNodeOneQP(Kokkos::RangePolicy(0, number_of_elements));
}

TEST(IntegrateMatrixTests, TestTwoElementsOneNodeOneQP_TeamPolicy) {
    constexpr auto number_of_elements = 2;

    TestTwoElementsOneNodeOneQP(Kokkos::TeamPolicy<>(number_of_elements, Kokkos::AUTO()));
}

TEST(IntegrateMatrixTests, TestTwoElementsOneNodeOneQP_ThreeIndices) {
    constexpr auto number_of_elements = 2;
    constexpr auto number_of_nodes = 1;

    TestTwoElementsOneNodeOneQP(Kokkos::MDRangePolicy{
        {0, 0, 0}, {number_of_elements, number_of_nodes, number_of_nodes}});
}

TEST(IntegrateMatrixTests, TestTwoElementsOneNodeOneQP_FourIndices) {
    constexpr auto number_of_elements = 2;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 1;

    TestTwoElementsOneNodeOneQP(Kokkos::MDRangePolicy{
        {0, 0, 0, 0}, {number_of_elements, number_of_nodes, number_of_nodes, number_of_qps}});
}

TEST(IntegrateMatrixTests, TestOneElementTwoNodesOneQP_OneIndex) {
    constexpr auto number_of_elements = 1;

    TestOneElementTwoNodesOneQP(Kokkos::RangePolicy(0, number_of_elements));
}

TEST(IntegrateMatrixTests, TestOneElementTwoNodesOneQP_TeamPolicy) {
    constexpr auto number_of_elements = 1;

    TestOneElementTwoNodesOneQP(Kokkos::TeamPolicy<>(number_of_elements, Kokkos::AUTO()));
}

TEST(IntegrateMatrixTests, TestOneElementTwoNodesOneQP_ThreeIndices) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 2;

    TestOneElementTwoNodesOneQP(Kokkos::MDRangePolicy{
        {0, 0, 0}, {number_of_elements, number_of_nodes, number_of_nodes}});
}

TEST(IntegrateMatrixTests, TestOneElementTwoNodesOneQP_FourIndices) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 2;
    constexpr auto number_of_qps = 1;

    TestOneElementTwoNodesOneQP(Kokkos::MDRangePolicy{
        {0, 0, 0, 0}, {number_of_elements, number_of_nodes, number_of_nodes, number_of_qps}});
}

TEST(IntegrateMatrixTests, TestOneElementOneNodeTwoQPs_OneIndex) {
    constexpr auto number_of_elements = 1;

    TestOneElementOneNodeTwoQPs(Kokkos::RangePolicy(0, number_of_elements));
}

TEST(IntegrateMatrixTests, TestOneElementOneNodeTwoQPs_TeamPolicy) {
    constexpr auto number_of_elements = 1;

    TestOneElementOneNodeTwoQPs(Kokkos::TeamPolicy<>(number_of_elements, Kokkos::AUTO()));
}

TEST(IntegrateMatrixTests, TestOneElementOneNodeTwoQPs_ThreeIndices) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;

    TestOneElementOneNodeTwoQPs(Kokkos::MDRangePolicy{
        {0, 0, 0}, {number_of_elements, number_of_nodes, number_of_nodes}});
}

TEST(IntegrateMatrixTests, TestOneElementOneNodeTwoQPs_FourIndices) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 2;

    TestOneElementOneNodeTwoQPs(Kokkos::MDRangePolicy{
        {0, 0, 0, 0}, {number_of_elements, number_of_nodes, number_of_nodes, number_of_qps}});
}
}  // namespace openturbine::restruct_poc::tests
