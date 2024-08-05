#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"

#include "src/restruct_poc/system/calculate_node_forces.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::tests {

TEST(CalculateNodeForcesTests, FE_OneNodeOneQP) {
    constexpr auto first_node = 0;
    constexpr auto first_qp = 0;
    constexpr auto num_nodes = 1;
    constexpr auto num_qps = 1;

    const auto weight = Kokkos::View<double[1][num_qps]>("weight");
    constexpr auto weight_data = std::array<double, 1>{2.};
    const auto weight_host =
        Kokkos::View<const double[1][num_qps], Kokkos::HostSpace>(weight_data.data());
    const auto weight_mirror = Kokkos::create_mirror(weight);
    Kokkos::deep_copy(weight_mirror, weight_host);
    Kokkos::deep_copy(weight, weight_mirror);

    const auto jacobian = Kokkos::View<double[1][num_qps]>("jacobian");
    constexpr auto jacobian_data = std::array{3.};
    const auto jacobian_host =
        Kokkos::View<const double[1][num_qps], Kokkos::HostSpace>(jacobian_data.data());
    const auto jacobian_mirror = Kokkos::create_mirror(jacobian);
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);

    const auto shape_interp = Kokkos::View<double[1][num_nodes][num_qps]>("shape_interp");
    constexpr auto shape_interp_data = std::array{4.};
    const auto shape_interp_host =
        Kokkos::View<const double[1][num_nodes][num_qps], Kokkos::HostSpace>(shape_interp_data.data()
        );
    const auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);

    const auto shape_deriv = Kokkos::View<double[1][num_nodes][num_qps]>("shape_deriv");
    constexpr auto shape_deriv_data = std::array{5.};
    const auto shape_deriv_host =
        Kokkos::View<const double[1][num_nodes][num_qps], Kokkos::HostSpace>(shape_deriv_data.data()
        );
    const auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);

    const auto Fc = Kokkos::View<double[num_qps][6]>("Fc");
    constexpr auto Fc_data = std::array{1., 2., 3., 4., 5., 6.};
    const auto Fc_host = Kokkos::View<const double[num_qps][6], Kokkos::HostSpace>(Fc_data.data());
    const auto Fc_mirror = Kokkos::create_mirror(Fc);
    Kokkos::deep_copy(Fc_mirror, Fc_host);
    Kokkos::deep_copy(Fc, Fc_mirror);

    const auto Fd = Kokkos::View<double[num_qps][6]>("Fd");
    constexpr auto Fd_data = std::array{7., 8., 9., 10., 11., 12.};
    const auto Fd_host = Kokkos::View<const double[num_qps][6], Kokkos::HostSpace>(Fd_data.data());
    const auto Fd_mirror = Kokkos::create_mirror(Fd);
    Kokkos::deep_copy(Fd_mirror, Fd_host);
    Kokkos::deep_copy(Fd, Fd_mirror);

    const auto FE = Kokkos::View<double[num_nodes][6]>("FE");

    Kokkos::parallel_for(
        "CalculateNodeForces_FE", num_nodes,
        CalculateNodeForces_FE{
            0, first_node, first_qp, num_qps, weight, jacobian, shape_interp, shape_deriv, Fc, Fd,
            FE}
    );

    constexpr auto FE_exact_data = std::array{178., 212., 246., 280., 314., 348.};
    const auto FE_exact =
        Kokkos::View<const double[num_nodes][6], Kokkos::HostSpace>(FE_exact_data.data());

    const auto FE_mirror = Kokkos::create_mirror(FE);
    Kokkos::deep_copy(FE_mirror, FE);
    CompareWithExpected(FE_mirror, FE_exact);
}

TEST(CalculateNodeForcesTests, FE_TwoNodesTwoQPs) {
    constexpr auto first_node = 0;
    constexpr auto first_qp = 0;
    constexpr auto num_nodes = 2;
    constexpr auto num_qps = 2;

    const auto weight = Kokkos::View<double[1][num_qps]>("weight");
    constexpr auto weight_data = std::array{2., 3.};
    const auto weight_host =
        Kokkos::View<const double[1][num_qps], Kokkos::HostSpace>(weight_data.data());
    const auto weight_mirror = Kokkos::create_mirror(weight);
    Kokkos::deep_copy(weight_mirror, weight_host);
    Kokkos::deep_copy(weight, weight_mirror);

    const auto jacobian = Kokkos::View<double[1][num_qps]>("jacobian");
    constexpr auto jacobian_data = std::array{4., 5.};
    const auto jacobian_host =
        Kokkos::View<const double[1][num_qps], Kokkos::HostSpace>(jacobian_data.data());
    const auto jacobian_mirror = Kokkos::create_mirror(jacobian);
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);

    const auto shape_interp = Kokkos::View<double[1][num_nodes][num_qps]>("shape_interp");
    constexpr auto shape_interp_data = std::array{6., 7., 8., 9.};
    const auto shape_interp_host =
        Kokkos::View<const double[1][num_nodes][num_qps], Kokkos::HostSpace>(shape_interp_data.data()
        );
    const auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);

    const auto shape_deriv = Kokkos::View<double[1][num_nodes][num_qps]>("shape_deriv");
    constexpr auto shape_deriv_data = std::array{10., 11., 12., 13.};
    const auto shape_deriv_host =
        Kokkos::View<const double[1][num_nodes][num_qps], Kokkos::HostSpace>(shape_deriv_data.data()
        );
    const auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);

    const auto Fc = Kokkos::View<double[num_qps][6]>("Fc");
    constexpr auto Fc_data = std::array{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    const auto Fc_host = Kokkos::View<const double[num_qps][6], Kokkos::HostSpace>(Fc_data.data());
    const auto Fc_mirror = Kokkos::create_mirror(Fc);
    Kokkos::deep_copy(Fc_mirror, Fc_host);
    Kokkos::deep_copy(Fc, Fc_mirror);

    const auto Fd = Kokkos::View<double[num_qps][6]>("Fd");
    constexpr auto Fd_data = std::array{13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.};
    const auto Fd_host = Kokkos::View<const double[num_qps][6], Kokkos::HostSpace>(Fd_data.data());
    const auto Fd_mirror = Kokkos::create_mirror(Fd);
    Kokkos::deep_copy(Fd_mirror, Fd_host);
    Kokkos::deep_copy(Fd, Fd_mirror);

    const auto FE = Kokkos::View<double[num_nodes][6]>("FE");

    Kokkos::parallel_for(
        "CalculateNodeForces_FE", num_nodes,
        CalculateNodeForces_FE{
            0, first_node, first_qp, num_qps, weight, jacobian, shape_interp, shape_deriv, Fc, Fd,
            FE}
    );

    constexpr auto FE_exact_data = std::array{2870., 3076., 3282., 3488., 3694., 3900.,
                                              3694., 3956., 4218., 4480., 4742., 5004.};
    const auto FE_exact =
        Kokkos::View<const double[num_nodes][6], Kokkos::HostSpace>(FE_exact_data.data());

    const auto FE_mirror = Kokkos::create_mirror(FE);
    Kokkos::deep_copy(FE_mirror, FE);
    CompareWithExpected(FE_mirror, FE_exact);
}

TEST(CalculateNodeForcesTests, FI_FG_OneNodeOneQP) {
    constexpr auto first_node = 0;
    constexpr auto first_qp = 0;
    constexpr auto num_nodes = 1;
    constexpr auto num_qps = 1;

    const auto weight = Kokkos::View<double[1][num_qps]>("weight");
    constexpr auto weight_data = std::array{2.};
    const auto weight_host =
        Kokkos::View<const double[1][num_qps], Kokkos::HostSpace>(weight_data.data());
    const auto weight_mirror = Kokkos::create_mirror(weight);
    Kokkos::deep_copy(weight_mirror, weight_host);
    Kokkos::deep_copy(weight, weight_mirror);

    const auto jacobian = Kokkos::View<double[1][num_qps]>("jacobian");
    constexpr auto jacobian_data = std::array{3.};
    const auto jacobian_host =
        Kokkos::View<const double[1][num_qps], Kokkos::HostSpace>(jacobian_data.data());
    const auto jacobian_mirror = Kokkos::create_mirror(jacobian);
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);

    const auto shape_interp = Kokkos::View<double[1][num_nodes][num_qps]>("shape_interp");
    constexpr auto shape_interp_data = std::array{4.};
    const auto shape_interp_host =
        Kokkos::View<const double[1][num_nodes][num_qps], Kokkos::HostSpace>(shape_interp_data.data()
        );
    const auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);

    const auto shape_deriv = Kokkos::View<double[1][num_nodes][num_qps]>("shape_deriv");
    constexpr auto shape_deriv_data = std::array{5.};
    const auto shape_deriv_host =
        Kokkos::View<const double[1][num_nodes][num_qps], Kokkos::HostSpace>(shape_deriv_data.data()
        );
    const auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);

    const auto Fig = Kokkos::View<double[num_qps][6]>("Fig");
    constexpr auto Fig_data = std::array{1., 2., 3., 4., 5., 6.};
    const auto Fig_host = Kokkos::View<const double[num_qps][6], Kokkos::HostSpace>(Fig_data.data());
    const auto Fig_mirror = Kokkos::create_mirror(Fig);
    Kokkos::deep_copy(Fig_mirror, Fig_host);
    Kokkos::deep_copy(Fig, Fig_mirror);

    const auto FIG = Kokkos::View<double[num_nodes][6]>("FIG");

    Kokkos::parallel_for(
        "CalculateNodeForces_FI_FG", num_nodes,
        CalculateNodeForces_FI_FG{
            0, first_node, first_qp, num_qps, weight, jacobian, shape_interp, shape_deriv, Fig, FIG}
    );

    constexpr auto FIG_exact_data = std::array{24., 48., 72., 96., 120., 144.};
    const auto FIG_exact =
        Kokkos::View<const double[num_nodes][6], Kokkos::HostSpace>(FIG_exact_data.data());

    const auto FIG_mirror = Kokkos::create_mirror(FIG);
    Kokkos::deep_copy(FIG_mirror, FIG);
    CompareWithExpected(FIG_mirror, FIG_exact);
}

TEST(CalculateNodeForcesTests, FI_FG_TwoNodesTwoQP) {
    constexpr auto first_node = 0;
    constexpr auto first_qp = 0;
    constexpr auto num_nodes = 2;
    constexpr auto num_qps = 2;

    const auto weight = Kokkos::View<double[1][num_qps]>("weight");
    constexpr auto weight_data = std::array{2., 3.};
    const auto weight_host =
        Kokkos::View<const double[1][num_qps], Kokkos::HostSpace>(weight_data.data());
    const auto weight_mirror = Kokkos::create_mirror(weight);
    Kokkos::deep_copy(weight_mirror, weight_host);
    Kokkos::deep_copy(weight, weight_mirror);

    const auto jacobian = Kokkos::View<double[1][num_qps]>("jacobian");
    constexpr auto jacobian_data = std::array{4., 5.};
    const auto jacobian_host =
        Kokkos::View<const double[1][num_qps], Kokkos::HostSpace>(jacobian_data.data());
    const auto jacobian_mirror = Kokkos::create_mirror(jacobian);
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);

    const auto shape_interp = Kokkos::View<double[1][num_nodes][num_qps]>("shape_interp");
    constexpr auto shape_interp_data = std::array{6., 7., 8., 9.};
    const auto shape_interp_host =
        Kokkos::View<const double[1][num_nodes][num_qps], Kokkos::HostSpace>(shape_interp_data.data()
        );
    const auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);

    const auto shape_deriv = Kokkos::View<double[1][num_nodes][num_qps]>("shape_deriv");
    constexpr auto shape_deriv_data = std::array{10., 11., 12., 13.};
    const auto shape_deriv_host =
        Kokkos::View<const double[1][num_nodes][num_qps], Kokkos::HostSpace>(shape_deriv_data.data()
        );
    const auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);

    const auto Fig = Kokkos::View<double[num_qps][6]>("Fig");
    constexpr auto Fig_data = std::array{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    const auto Fig_host = Kokkos::View<const double[num_qps][6], Kokkos::HostSpace>(Fig_data.data());
    const auto Fig_mirror = Kokkos::create_mirror(Fig);
    Kokkos::deep_copy(Fig_mirror, Fig_host);
    Kokkos::deep_copy(Fig, Fig_mirror);

    const auto FIG = Kokkos::View<double[num_nodes][6]>("FIG");

    Kokkos::parallel_for(
        "CalculateNodeForces_FI_FG", num_nodes,
        CalculateNodeForces_FI_FG{
            0, first_node, first_qp, num_qps, weight, jacobian, shape_interp, shape_deriv, Fig, FIG}
    );

    constexpr auto FIG_exact_data =
        std::array{783., 936., 1089., 1242., 1395., 1548., 1009., 1208., 1407., 1606., 1805., 2004.};
    const auto FIG_exact =
        Kokkos::View<const double[num_nodes][6], Kokkos::HostSpace>(FIG_exact_data.data());

    const auto FIG_mirror = Kokkos::create_mirror(FIG);
    Kokkos::deep_copy(FIG_mirror, FIG);
    CompareWithExpected(FIG_mirror, FIG_exact);
}

}  // namespace openturbine::tests