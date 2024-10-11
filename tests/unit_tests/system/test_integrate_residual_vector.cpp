#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"
#include "test_integrate_matrix.hpp"

#include "src/beams/beams.hpp"
#include "src/system/integrate_residual_vector.hpp"

namespace openturbine::tests {

TEST(IntegrateResidualVector, OneElementOneNodeOneQP_Fc) {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({0.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({0.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({3.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = get_qp_Fc<number_of_qps>({1., 2., 3., 4., 5., 6.});
    const auto qp_Fd = QpVectorView("qp_Fd");
    const auto qp_Fi = QpVectorView("qp_Fi");
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, number_of_nodes * 6>{6., 12., 18., 24., 30., 36.};
    const auto resid_exact =
        Kokkos::View<const double[1][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

TEST(IntegrateResidualVector, OneElementOneNodeOneQP_Fd) {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({3.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({4.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({0.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = QpVectorView("qp_Fc");
    const auto qp_Fd = get_qp_Fd<number_of_qps>({1., 2., 3., 4., 5., 6.});
    const auto qp_Fi = QpVectorView("qp_Fi");
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, number_of_nodes * 6>{24., 48., 72., 96., 120., 144.};
    const auto resid_exact =
        Kokkos::View<const double[1][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

TEST(IntegrateResidualVector, OneElementOneNodeOneQP_Fi) {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({3.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({4.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({0.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = QpVectorView("qp_Fc");
    const auto qp_Fd = QpVectorView("qp_Fd");
    const auto qp_Fi = get_qp_Fi<number_of_qps>({1., 2., 3., 4., 5., 6.});
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, number_of_nodes * 6>{24., 48., 72., 96., 120., 144.};
    const auto resid_exact =
        Kokkos::View<const double[1][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

TEST(IntegrateResidualVector, OneElementOneNodeOneQP_Fe) {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({3.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({4.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({0.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = QpVectorView("qp_Fc");
    const auto qp_Fd = QpVectorView("qp_Fd");
    const auto qp_Fi = QpVectorView("qp_Fi");
    const auto qp_Fe = get_qp_Fe<number_of_qps>({1., 2., 3., 4., 5., 6.});
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, number_of_nodes * 6>{-24., -48., -72., -96., -120., -144.};
    const auto resid_exact =
        Kokkos::View<const double[1][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

TEST(IntegrateResidualVector, OneElementOneNodeOneQP_Fg) {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({3.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({4.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({0.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = QpVectorView("qp_Fc");
    const auto qp_Fd = QpVectorView("qp_Fd");
    const auto qp_Fi = QpVectorView("qp_Fi");
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = get_qp_Fg<number_of_qps>({1., 2., 3., 4., 5., 6.});

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, number_of_nodes * 6>{-24., -48., -72., -96., -120., -144.};
    const auto resid_exact =
        Kokkos::View<const double[1][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

TEST(IntegrateResidualVector, OneElementOneNodeOneQP_FX) {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({3.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({4.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({0.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = get_node_FX<number_of_nodes>({1., 2., 3., 4., 5., 6.});
    const auto qp_Fc = QpVectorView("qp_Fc");
    const auto qp_Fd = QpVectorView("qp_Fd");
    const auto qp_Fi = QpVectorView("qp_Fi");
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, number_of_nodes * 6>{-1., -2., -3., -4., -5., -6.};
    const auto resid_exact =
        Kokkos::View<const double[1][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

TEST(IntegrateResidualVector, TwoElementsOneNodeOneQP) {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({0.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({0.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({3.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc_1 = get_qp_Fc<number_of_qps>({1., 2., 3., 4., 5., 6.});
    const auto qp_Fc_2 = get_qp_Fc<number_of_qps>({2., 4., 6., 8., 10., 12.});
    const auto qp_Fd = QpVectorView("qp_Fd");
    const auto qp_Fi = QpVectorView("qp_Fi");
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[2][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc_1,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            1U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc_2,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, 2 * number_of_nodes * 6>{6.,  12., 18., 24., 30., 36.,
                                                    12., 24., 36., 48., 60., 72.};
    const auto resid_exact =
        Kokkos::View<const double[2][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

TEST(IntegrateResidualVector, OneElementOneNodeTwoQPs) {
    constexpr auto number_of_nodes = size_t{1U};
    constexpr auto number_of_qps = size_t{2U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2., 1.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({0.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({0.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({3., 2.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = get_qp_Fc<number_of_qps>({1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    const auto qp_Fd = QpVectorView("qp_Fd");
    const auto qp_Fi = QpVectorView("qp_Fi");
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, number_of_nodes * 6>{20., 28., 36., 44., 52., 60.};
    const auto resid_exact =
        Kokkos::View<const double[1][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

TEST(IntegrateResidualVector, OneElementTwoNodesOneQP) {
    constexpr auto number_of_nodes = size_t{2U};
    constexpr auto number_of_qps = size_t{1U};

    const auto qp_weights = get_qp_weights<number_of_qps>({2.});
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({0.});
    const auto shape_interp_left = get_shape_interp<number_of_nodes, number_of_qps>({0.});
    const auto shape_interp = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_interp");
    Kokkos::deep_copy(shape_interp, shape_interp_left);
    const auto shape_deriv_left = get_shape_interp_deriv<number_of_nodes, number_of_qps>({3.});
    const auto shape_deriv = Kokkos::View<double[number_of_nodes][number_of_qps]>("shape_deriv");
    Kokkos::deep_copy(shape_deriv, shape_deriv_left);
    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = get_qp_Fc<number_of_qps>({1., 2., 3., 4., 5., 6.});
    const auto qp_Fd = QpVectorView("qp_Fd");
    const auto qp_Fi = QpVectorView("qp_Fi");
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", number_of_nodes,
        IntegrateResidualVectorElement{
            0U, number_of_qps, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc,
            qp_Fd, qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    constexpr auto resid_exact_data =
        std::array<double, number_of_nodes * 6>{6., 12., 18., 24., 30., 36.};
    const auto resid_exact =
        Kokkos::View<const double[1][number_of_nodes][6], Kokkos::HostSpace>(resid_exact_data.data()
        );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact);
}

}  // namespace openturbine::tests
