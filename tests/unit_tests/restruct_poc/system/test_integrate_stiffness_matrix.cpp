#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"
#include "test_integrate_matrix.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/integrate_stiffness_matrix.hpp"
#include "src/restruct_poc/types.hpp"
#include "tests/unit_tests/restruct_poc/test_utilities.hpp"

namespace openturbine::tests {

void TestIntegrateStiffnessMatrix_1Element1Node1QP(
    const Kokkos::View<const double[1][6][6]>& qp_Kuu,
    const Kokkos::View<const double[1][6][6]>& qp_Puu,
    const Kokkos::View<const double[1][6][6]>& qp_Cuu,
    const Kokkos::View<const double[1][6][6]>& qp_Ouu,
    const Kokkos::View<const double[1][6][6]>& qp_Quu, const std::array<double, 36>& exact_M_data
) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 1;

    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({3.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({4.});
    const auto shape_interp_deriv =
        get_shape_interp_deriv<number_of_elements, number_of_nodes, number_of_qps>({5.});

    auto gbl_M = Kokkos::View<double[1][6][6]>("global_M");

    const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
    const auto integrator = IntegrateStiffnessMatrixElement{0,
                                                            number_of_qps,
                                                            0,
                                                            0,
                                                            qp_weights,
                                                            qp_jacobian,
                                                            shape_interp,
                                                            shape_interp_deriv,
                                                            qp_Kuu,
                                                            qp_Puu,
                                                            qp_Cuu,
                                                            qp_Ouu,
                                                            qp_Quu,
                                                            gbl_M};
    Kokkos::parallel_for(policy, integrator);

    const auto exact_M = Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

void TestIntegrateStiffnessMatrix_1Element1Node1QP_Kuu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = get_qp_Kuu<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003., 1004., 1005., 1006.,
         2001., 2002., 2003., 2004., 2005., 2006., 3001., 3002., 3003., 3004., 3005., 3006.,
         4001., 4002., 4003., 4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.}
    );
    const auto qp_Puu = QpMatrixView("Puu");
    const auto qp_Quu = QpMatrixView("Quu");
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = QpMatrixView("Ouu");

    constexpr auto exact_M_data =
        std::array{000096., 000192., 000288., 000384., 000480., 000576., 096096., 096192., 096288.,
                   096384., 096480., 096576., 192096., 192192., 192288., 192384., 192480., 192576.,
                   288096., 288192., 288288., 288384., 288480., 288576., 384096., 384192., 384288.,
                   384384., 384480., 384576., 480096., 480192., 480288., 480384., 480480., 480576.};

    TestIntegrateStiffnessMatrix_1Element1Node1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementOneNodeOneQP_Kuu) {
    TestIntegrateStiffnessMatrix_1Element1Node1QP_Kuu();
}

void TestIntegrateStiffnessMatrix_1Element1Node1QP_Puu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Puu = get_qp_Puu<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003., 1004., 1005., 1006.,
         2001., 2002., 2003., 2004., 2005., 2006., 3001., 3002., 3003., 3004., 3005., 3006.,
         4001., 4002., 4003., 4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.}
    );
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = QpMatrixView("Ouu");
    const auto qp_Quu = QpMatrixView("Quu");

    constexpr auto exact_M_data =
        std::array{000040., 000080., 000120., 000160., 000200., 000240., 040040., 040080., 040120.,
                   040160., 040200., 040240., 080040., 080080., 080120., 080160., 080200., 080240.,
                   120040., 120080., 120120., 120160., 120200., 120240., 160040., 160080., 160120.,
                   160160., 160200., 160240., 200040., 200080., 200120., 200160., 200200., 200240.};

    TestIntegrateStiffnessMatrix_1Element1Node1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementOneNodeOneQP_Puu) {
    TestIntegrateStiffnessMatrix_1Element1Node1QP_Puu();
}

void TestIntegrateStiffnessMatrix_1Element1Node1QP_Quu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Puu = QpMatrixView("Puu");
    const auto qp_Quu = get_qp_Quu<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 1001., 1002., 1003., 1004., 1005., 1006.,
         2001., 2002., 2003., 2004., 2005., 2006., 3001., 3002., 3003., 3004., 3005., 3006.,
         4001., 4002., 4003., 4004., 4005., 4006., 5001., 5002., 5003., 5004., 5005., 5006.}
    );
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = QpMatrixView("Ouu");

    constexpr auto exact_M_data =
        std::array{000096., 000192., 000288., 000384., 000480., 000576., 096096., 096192., 096288.,
                   096384., 096480., 096576., 192096., 192192., 192288., 192384., 192480., 192576.,
                   288096., 288192., 288288., 288384., 288480., 288576., 384096., 384192., 384288.,
                   384384., 384480., 384576., 480096., 480192., 480288., 480384., 480480., 480576.};

    TestIntegrateStiffnessMatrix_1Element1Node1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementOneNodeOneQP_Quu) {
    TestIntegrateStiffnessMatrix_1Element1Node1QP_Quu();
}

void TestIntegrateStiffnessMatrix_1Element1Node1QP_Cuu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Puu = QpMatrixView("Puu");
    const auto qp_Quu = QpMatrixView("Quu");
    const auto qp_Cuu = get_qp_Cuu<number_of_elements, number_of_qps>(
        {03003., 03006., 03009., 03012., 03015., 03018., 06003., 06006., 06009.,
         06012., 06015., 06018., 09003., 09006., 09009., 09012., 09015., 09018.,
         12003., 12006., 12009., 12012., 12015., 12018., 15003., 15006., 15009.,
         15012., 15015., 15018., 18003., 18006., 18009., 18012., 18015., 18018.}
    );
    const auto qp_Ouu = QpMatrixView("Ouu");

    constexpr auto exact_M_data =
        std::array{050050., 050100., 050150., 050200., 050250., 050300., 100050., 100100., 100150.,
                   100200., 100250., 100300., 150050., 150100., 150150., 150200., 150250., 150300.,
                   200050., 200100., 200150., 200200., 200250., 200300., 250050., 250100., 250150.,
                   250200., 250250., 250300., 300050., 300100., 300150., 300200., 300250., 300300.};

    TestIntegrateStiffnessMatrix_1Element1Node1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementOneNodeOneQP_Cuu) {
    TestIntegrateStiffnessMatrix_1Element1Node1QP_Cuu();
}

void TestIntegrateStiffnessMatrix_1Element1Node1QP_Ouu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Puu = QpMatrixView("Puu");
    const auto qp_Quu = QpMatrixView("Quu");
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = get_qp_Ouu<number_of_elements, number_of_qps>(
        {1001., 1002., 1003., 1004., 1005., 1006., 2001., 2002., 2003., 2004., 2005., 2006.,
         3001., 3002., 3003., 3004., 3005., 3006., 4001., 4002., 4003., 4004., 4005., 4006.,
         5001., 5002., 5003., 5004., 5005., 5006., 6001., 6002., 6003., 6004., 6005., 6006.}
    );

    constexpr auto exact_M_data =
        std::array{040040., 040080., 040120., 040160., 040200., 040240., 080040., 080080., 080120.,
                   080160., 080200., 080240., 120040., 120080., 120120., 120160., 120200., 120240.,
                   160040., 160080., 160120., 160160., 160200., 160240., 200040., 200080., 200120.,
                   200160., 200200., 200240., 240040., 240080., 240120., 240160., 240200., 240240.};

    TestIntegrateStiffnessMatrix_1Element1Node1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementOneNodeOneQP_Ouu) {
    TestIntegrateStiffnessMatrix_1Element1Node1QP_Ouu();
}

void TestIntegrateStiffnessMatrix_2Elements1Node1QP() {
    constexpr auto number_of_elements = 2;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 1;

    const auto element_indices =
        get_element_indices<number_of_elements, number_of_nodes, number_of_qps>();
    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({1., 1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({1., 1.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({1., 0., 1., 0.});
    const auto shape_interp_deriv =
        get_shape_interp_deriv<number_of_elements, number_of_nodes, number_of_qps>({1., 0., 1., 0.});

    const auto qp_Puu = get_qp_Puu<number_of_elements, number_of_qps>(
        {00001., 00002., 00003., 00004., 00005., 00006., 00101., 00102., 00103.,
         00104., 00105., 00106., 00201., 00202., 00203., 00204., 00205., 00206.,
         00301., 00302., 00303., 00304., 00305., 00306., 00401., 00402., 00403.,
         00404., 00405., 00406., 00501., 00502., 00503., 00504., 00505., 00506.,

         00001., 00002., 00003., 00004., 00005., 00006., 10001., 10002., 10003.,
         10004., 10005., 10006., 20001., 20002., 20003., 20004., 20005., 20006.,
         30001., 30002., 30003., 30004., 30005., 30006., 40001., 40002., 40003.,
         40004., 40005., 40006., 50001., 50002., 50003., 50004., 50005., 50006.}
    );

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Quu = QpMatrixView("Quu");
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = QpMatrixView("Ouu");

    auto gbl_M = Kokkos::View<double[2][6][6]>("global_M");

    {
        const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
        const auto integrator = IntegrateStiffnessMatrixElement{0,
                                                                number_of_qps,
                                                                0,
                                                                0,
                                                                qp_weights,
                                                                qp_jacobian,
                                                                shape_interp,
                                                                shape_interp_deriv,
                                                                qp_Kuu,
                                                                qp_Puu,
                                                                qp_Cuu,
                                                                qp_Ouu,
                                                                qp_Quu,
                                                                gbl_M};
        Kokkos::parallel_for(policy, integrator);
    }

    {
        const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
        const auto integrator = IntegrateStiffnessMatrixElement{1,
                                                                number_of_qps,
                                                                1,
                                                                1,
                                                                qp_weights,
                                                                qp_jacobian,
                                                                shape_interp,
                                                                shape_interp_deriv,
                                                                qp_Kuu,
                                                                qp_Puu,
                                                                qp_Cuu,
                                                                qp_Ouu,
                                                                qp_Quu,
                                                                gbl_M};
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

    const auto exact_M = Kokkos::View<const double[2][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

TEST(IntegrateStiffnessMatrixTests, TwoElementsOneNodeOneQP) {
    TestIntegrateStiffnessMatrix_2Elements1Node1QP();
}

void TestIntegrateStiffnessMatrix_1Element2Nodes1QP(
    const Kokkos::View<const double[1][6][6]>& qp_Kuu,
    const Kokkos::View<const double[1][6][6]>& qp_Puu,
    const Kokkos::View<const double[1][6][6]>& qp_Cuu,
    const Kokkos::View<const double[1][6][6]>& qp_Ouu,
    const Kokkos::View<const double[1][6][6]>& qp_Quu, const std::array<double, 144>& exact_M_data
) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 2;
    constexpr auto number_of_qps = 1;

    const auto element_indices =
        get_element_indices<number_of_elements, number_of_nodes, number_of_qps>();
    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({1.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({1., 2.});
    const auto shape_interp_deriv =
        get_shape_interp_deriv<number_of_elements, number_of_nodes, number_of_qps>({1., 4.});

    auto gbl_M = Kokkos::View<double[1][12][12]>("global_M");

    const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
    const auto integrator = IntegrateStiffnessMatrixElement{0,
                                                            number_of_qps,
                                                            0,
                                                            0,
                                                            qp_weights,
                                                            qp_jacobian,
                                                            shape_interp,
                                                            shape_interp_deriv,
                                                            qp_Kuu,
                                                            qp_Puu,
                                                            qp_Cuu,
                                                            qp_Ouu,
                                                            qp_Quu,
                                                            gbl_M};
    Kokkos::parallel_for(policy, integrator);

    const auto exact_M =
        Kokkos::View<const double[1][12][12], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

void TestIntegrateStiffnessMatrix_1Element2Nodes1QP_Puu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Puu = get_qp_Puu<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 0101., 0102., 0103., 0104., 0105., 0106.,
         0201., 0202., 0203., 0204., 0205., 0206., 0301., 0302., 0303., 0304., 0305., 0306.,
         0401., 0402., 0403., 0404., 0405., 0406., 0501., 0502., 0503., 0504., 0505., 0506.}
    );
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Quu = QpMatrixView("Quu");
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = QpMatrixView("Ouu");

    constexpr auto exact_M_data = std::array{
        00001., 00002., 00003., 00004., 00005., 00006., 00004., 00008., 00012., 00016., 00020.,
        00024., 00101., 00102., 00103., 00104., 00105., 00106., 00404., 00408., 00412., 00416.,
        00420., 00424., 00201., 00202., 00203., 00204., 00205., 00206., 00804., 00808., 00812.,
        00816., 00820., 00824., 00301., 00302., 00303., 00304., 00305., 00306., 01204., 01208.,
        01212., 01216., 01220., 01224., 00401., 00402., 00403., 00404., 00405., 00406., 01604.,
        01608., 01612., 01616., 01620., 01624., 00501., 00502., 00503., 00504., 00505., 00506.,
        02004., 02008., 02012., 02016., 02020., 02024.,

        00002., 00004., 00006., 00008., 00010., 00012., 00008., 00016., 00024., 00032., 00040.,
        00048., 00202., 00204., 00206., 00208., 00210., 00212., 00808., 00816., 00824., 00832.,
        00840., 00848., 00402., 00404., 00406., 00408., 00410., 00412., 01608., 01616., 01624.,
        01632., 01640., 01648., 00602., 00604., 00606., 00608., 00610., 00612., 02408., 02416.,
        02424., 02432., 02440., 02448., 00802., 00804., 00806., 00808., 00810., 00812., 03208.,
        03216., 03224., 03232., 03240., 03248., 01002., 01004., 01006., 01008., 01010., 01012.,
        04008., 04016., 04024., 04032., 04040., 04048.};

    TestIntegrateStiffnessMatrix_1Element2Nodes1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementTwoNodesOneQP_Puu) {
    TestIntegrateStiffnessMatrix_1Element2Nodes1QP_Puu();
}

void TestIntegrateStiffnessMatrix_1Element2Nodes1QP_Quu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Puu = QpMatrixView("Puu");
    const auto qp_Quu = get_qp_Quu<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 0101., 0102., 0103., 0104., 0105., 0106.,
         0201., 0202., 0203., 0204., 0205., 0206., 0301., 0302., 0303., 0304., 0305., 0306.,
         0401., 0402., 0403., 0404., 0405., 0406., 0501., 0502., 0503., 0504., 0505., 0506.}
    );
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = QpMatrixView("Ouu");

    constexpr auto exact_M_data = std::array{
        00001., 00002., 00003., 00004., 00005., 00006., 00002., 00004., 00006., 00008., 00010.,
        00012., 00101., 00102., 00103., 00104., 00105., 00106., 00202., 00204., 00206., 00208.,
        00210., 00212., 00201., 00202., 00203., 00204., 00205., 00206., 00402., 00404., 00406.,
        00408., 00410., 00412., 00301., 00302., 00303., 00304., 00305., 00306., 00602., 00604.,
        00606., 00608., 00610., 00612., 00401., 00402., 00403., 00404., 00405., 00406., 00802.,
        00804., 00806., 00808., 00810., 00812., 00501., 00502., 00503., 00504., 00505., 00506.,
        01002., 01004., 01006., 01008., 01010., 01012.,

        00002., 00004., 00006., 00008., 00010., 00012., 00004., 00008., 00012., 00016., 00020.,
        00024., 00202., 00204., 00206., 00208., 00210., 00212., 00404., 00408., 00412., 00416.,
        00420., 00424., 00402., 00404., 00406., 00408., 00410., 00412., 00804., 00808., 00812.,
        00816., 00820., 00824., 00602., 00604., 00606., 00608., 00610., 00612., 01204., 01208.,
        01212., 01216., 01220., 01224., 00802., 00804., 00806., 00808., 00810., 00812., 01604.,
        01608., 01612., 01616., 01620., 01624., 01002., 01004., 01006., 01008., 01010., 01012.,
        02004., 02008., 02012., 02016., 02020., 02024.};

    TestIntegrateStiffnessMatrix_1Element2Nodes1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementTwoNodesOneQP_Quu) {
    TestIntegrateStiffnessMatrix_1Element2Nodes1QP_Quu();
}

void TestIntegrateStiffnessMatrix_1Element2Nodes1QP_Cuu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Puu = QpMatrixView("Puu");
    const auto qp_Quu = QpMatrixView("Quu");
    const auto qp_Cuu = get_qp_Cuu<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 0101., 0102., 0103., 0104., 0105., 0106.,
         0201., 0202., 0203., 0204., 0205., 0206., 0301., 0302., 0303., 0304., 0305., 0306.,
         0401., 0402., 0403., 0404., 0405., 0406., 0501., 0502., 0503., 0504., 0505., 0506.}
    );
    const auto qp_Ouu = QpMatrixView("Ouu");

    constexpr auto exact_M_data = std::array{
        00001., 00002., 00003., 00004., 00005., 00006., 00004., 00008., 00012., 00016., 00020.,
        00024., 00101., 00102., 00103., 00104., 00105., 00106., 00404., 00408., 00412., 00416.,
        00420., 00424., 00201., 00202., 00203., 00204., 00205., 00206., 00804., 00808., 00812.,
        00816., 00820., 00824., 00301., 00302., 00303., 00304., 00305., 00306., 01204., 01208.,
        01212., 01216., 01220., 01224., 00401., 00402., 00403., 00404., 00405., 00406., 01604.,
        01608., 01612., 01616., 01620., 01624., 00501., 00502., 00503., 00504., 00505., 00506.,
        02004., 02008., 02012., 02016., 02020., 02024.,

        00004., 00008., 00012., 00016., 00020., 00024., 00016., 00032., 00048., 00064., 00080.,
        00096., 00404., 00408., 00412., 00416., 00420., 00424., 01616., 01632., 01648., 01664.,
        01680., 01696., 00804., 00808., 00812., 00816., 00820., 00824., 03216., 03232., 03248.,
        03264., 03280., 03296., 01204., 01208., 01212., 01216., 01220., 01224., 04816., 04832.,
        04848., 04864., 04880., 04896., 01604., 01608., 01612., 01616., 01620., 01624., 06416.,
        06432., 06448., 06464., 06480., 06496., 02004., 02008., 02012., 02016., 02020., 02024.,
        08016., 08032., 08048., 08064., 08080., 08096.};

    TestIntegrateStiffnessMatrix_1Element2Nodes1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementTwoNodesOneQP_Cuu) {
    TestIntegrateStiffnessMatrix_1Element2Nodes1QP_Cuu();
}

void TestIntegrateStiffnessMatrix_1Element2Nodes1QP_Ouu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 1;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Puu = QpMatrixView("Puu");
    const auto qp_Quu = QpMatrixView("Quu");
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = get_qp_Ouu<number_of_elements, number_of_qps>(
        {0001., 0002., 0003., 0004., 0005., 0006., 0101., 0102., 0103., 0104., 0105., 0106.,
         0201., 0202., 0203., 0204., 0205., 0206., 0301., 0302., 0303., 0304., 0305., 0306.,
         0401., 0402., 0403., 0404., 0405., 0406., 0501., 0502., 0503., 0504., 0505., 0506.}
    );

    constexpr auto exact_M_data = std::array{
        00001., 00002., 00003., 00004., 00005., 00006., 00002., 00004., 00006., 00008., 00010.,
        00012., 00101., 00102., 00103., 00104., 00105., 00106., 00202., 00204., 00206., 00208.,
        00210., 00212., 00201., 00202., 00203., 00204., 00205., 00206., 00402., 00404., 00406.,
        00408., 00410., 00412., 00301., 00302., 00303., 00304., 00305., 00306., 00602., 00604.,
        00606., 00608., 00610., 00612., 00401., 00402., 00403., 00404., 00405., 00406., 00802.,
        00804., 00806., 00808., 00810., 00812., 00501., 00502., 00503., 00504., 00505., 00506.,
        01002., 01004., 01006., 01008., 01010., 01012.,

        00004., 00008., 00012., 00016., 00020., 00024., 00008., 00016., 00024., 00032., 00040.,
        00048., 00404., 00408., 00412., 00416., 00420., 00424., 00808., 00816., 00824., 00832.,
        00840., 00848., 00804., 00808., 00812., 00816., 00820., 00824., 01608., 01616., 01624.,
        01632., 01640., 01648., 01204., 01208., 01212., 01216., 01220., 01224., 02408., 02416.,
        02424., 02432., 02440., 02448., 01604., 01608., 01612., 01616., 01620., 01624., 03208.,
        03216., 03224., 03232., 03240., 03248., 02004., 02008., 02012., 02016., 02020., 02024.,
        04008., 04016., 04024., 04032., 04040., 04048.};

    TestIntegrateStiffnessMatrix_1Element2Nodes1QP(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementTwoNodesOneQP_Ouu) {
    TestIntegrateStiffnessMatrix_1Element2Nodes1QP_Ouu();
}

void TestIntegrateStiffnessMatrix_1Element1Node2QPs(
    const Kokkos::View<const double[2][6][6]>& qp_Kuu,
    const Kokkos::View<const double[2][6][6]>& qp_Puu,
    const Kokkos::View<const double[2][6][6]>& qp_Cuu,
    const Kokkos::View<const double[2][6][6]>& qp_Ouu,
    const Kokkos::View<const double[2][6][6]>& qp_Quu, const std::array<double, 36>& exact_M_data
) {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_nodes = 1;
    constexpr auto number_of_qps = 2;

    const auto element_indices =
        get_element_indices<number_of_elements, number_of_nodes, number_of_qps>();
    const auto qp_weights = get_qp_weights<number_of_elements, number_of_qps>({1., 3.});
    const auto qp_jacobian = get_qp_jacobian<number_of_elements, number_of_qps>({2., 4.});
    const auto shape_interp =
        get_shape_interp<number_of_elements, number_of_nodes, number_of_qps>({1., 1.});
    const auto shape_interp_deriv =
        get_shape_interp_deriv<number_of_elements, number_of_nodes, number_of_qps>({1., 1.});

    auto gbl_M = Kokkos::View<double[1][6][6]>("global_M");

    const auto policy = Kokkos::MDRangePolicy({0, 0}, {number_of_nodes, number_of_nodes});
    const auto integrator = IntegrateStiffnessMatrixElement{0,
                                                            number_of_qps,
                                                            0,
                                                            0,
                                                            qp_weights,
                                                            qp_jacobian,
                                                            shape_interp,
                                                            shape_interp_deriv,
                                                            qp_Kuu,
                                                            qp_Puu,
                                                            qp_Cuu,
                                                            qp_Ouu,
                                                            qp_Quu,
                                                            gbl_M};
    Kokkos::parallel_for(policy, integrator);

    const auto exact_M = Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(exact_M_data.data());

    auto gbl_M_mirror = Kokkos::create_mirror(gbl_M);
    Kokkos::deep_copy(gbl_M_mirror, gbl_M);
    CompareWithExpected(gbl_M_mirror, exact_M);
}

void TestIntegrateStiffnessMatrix_1Element1Node2QPs_Puu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 2;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Puu = get_qp_Puu<number_of_elements, number_of_qps>(
        {10001., 10002., 10003., 10004., 10005., 10006., 10101., 10102., 10103.,
         10104., 10105., 10106., 10201., 10202., 10203., 10204., 10205., 10206.,
         10301., 10302., 10303., 10304., 10305., 10306., 10401., 10402., 10403.,
         10404., 10405., 10406., 10501., 10502., 10503., 10504., 10505., 10506.,

         20001., 20002., 20003., 20004., 20005., 20006., 20101., 20102., 20103.,
         20104., 20105., 20106., 20201., 20202., 20203., 20204., 20205., 20206.,
         20301., 20302., 20303., 20304., 20305., 20306., 20401., 20402., 20403.,
         20404., 20405., 20406., 20501., 20502., 20503., 20504., 20505., 20506.}
    );
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Quu = QpMatrixView("Quu");
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = QpMatrixView("Ouu");

    constexpr auto exact_M_data =
        std::array{70004., 70008., 70012., 70016., 70020., 70024., 70404., 70408., 70412.,
                   70416., 70420., 70424., 70804., 70808., 70812., 70816., 70820., 70824.,
                   71204., 71208., 71212., 71216., 71220., 71224., 71604., 71608., 71612.,
                   71616., 71620., 71624., 72004., 72008., 72012., 72016., 72020., 72024.};

    TestIntegrateStiffnessMatrix_1Element1Node2QPs(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementOneNodeTwoQPs_Puu) {
    TestIntegrateStiffnessMatrix_1Element1Node2QPs_Puu();
}

void TestIntegrateStiffnessMatrix_1Element1Node2QPs_Quu() {
    constexpr auto number_of_elements = 1;
    constexpr auto number_of_qps = 2;

    using QpMatrixView = Kokkos::View<double[number_of_elements * number_of_qps][6][6]>;
    const auto qp_Kuu = QpMatrixView("Kuu");
    const auto qp_Puu = QpMatrixView("Cuu");
    const auto qp_Quu = get_qp_Quu<number_of_elements, number_of_qps>(
        {10001., 10002., 10003., 10004., 10005., 10006., 10101., 10102., 10103.,
         10104., 10105., 10106., 10201., 10202., 10203., 10204., 10205., 10206.,
         10301., 10302., 10303., 10304., 10305., 10306., 10401., 10402., 10403.,
         10404., 10405., 10406., 10501., 10502., 10503., 10504., 10505., 10506.,

         20001., 20002., 20003., 20004., 20005., 20006., 20101., 20102., 20103.,
         20104., 20105., 20106., 20201., 20202., 20203., 20204., 20205., 20206.,
         20301., 20302., 20303., 20304., 20305., 20306., 20401., 20402., 20403.,
         20404., 20405., 20406., 20501., 20502., 20503., 20504., 20505., 20506.}
    );
    const auto qp_Cuu = QpMatrixView("Cuu");
    const auto qp_Ouu = QpMatrixView("Ouu");

    constexpr auto exact_M_data =
        std::array{260014., 260028., 260042., 260056., 260070., 260084., 261414., 261428., 261442.,
                   261456., 261470., 261484., 262814., 262828., 262842., 262856., 262870., 262884.,
                   264214., 264228., 264242., 264256., 264270., 264284., 265614., 265628., 265642.,
                   265656., 265670., 265684., 267014., 267028., 267042., 267056., 267070., 267084.};

    TestIntegrateStiffnessMatrix_1Element1Node2QPs(
        qp_Kuu, qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, exact_M_data
    );
}

TEST(IntegrateStiffnessMatrixTests, OneElementOneNodeTwoQPs_Quu) {
    TestIntegrateStiffnessMatrix_1Element1Node2QPs_Quu();
}

}  // namespace openturbine::tests