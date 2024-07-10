#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_node_forces.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateNodeForcesTests, FE_OneNodeOneQP) {
    constexpr auto first_node = 0;
    constexpr auto first_qp = 0;
    constexpr auto num_nodes = 1;
    constexpr auto num_qps = 1;

    auto weight = Kokkos::View<double[num_qps]>("weight");
    auto weight_data = std::array<double, 1>{2.};
    auto weight_host = Kokkos::View<double[num_qps], Kokkos::HostSpace>(weight_data.data());
    auto weight_mirror = Kokkos::create_mirror(weight);
    Kokkos::deep_copy(weight_mirror, weight_host);
    Kokkos::deep_copy(weight, weight_mirror);

    auto jacobian = Kokkos::View<double[num_qps]>("jacobian");
    auto jacobian_data = std::array<double, 1>{3.};
    auto jacobian_host = Kokkos::View<double[num_qps], Kokkos::HostSpace>(jacobian_data.data());
    auto jacobian_mirror = Kokkos::create_mirror(jacobian);
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);

    auto shape_interp = Kokkos::View<double[num_nodes][num_qps]>("shape_interp");
    auto shape_interp_data = std::array<double, 1>{4.};
    auto shape_interp_host = Kokkos::View<double[num_nodes][num_qps], Kokkos::HostSpace>(shape_interp_data.data());
    auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);

    auto shape_deriv = Kokkos::View<double[num_nodes][num_qps]>("shape_deriv");
    auto shape_deriv_data = std::array<double, 1>{5.};
    auto shape_deriv_host = Kokkos::View<double[num_nodes][num_qps], Kokkos::HostSpace>(shape_deriv_data.data());
    auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);

    auto Fc = Kokkos::View<double[num_qps][6]>("Fc");
    auto Fc_data = std::array<double, 6>{1., 2., 3., 4., 5., 6.};
    auto Fc_host = Kokkos::View<double[num_qps][6], Kokkos::HostSpace>(Fc_data.data());
    auto Fc_mirror = Kokkos::create_mirror(Fc);
    Kokkos::deep_copy(Fc_mirror, Fc_host);
    Kokkos::deep_copy(Fc, Fc_mirror);

    auto Fd = Kokkos::View<double[num_qps][6]>("Fd");
    auto Fd_data = std::array<double, 6>{7., 8., 9., 10., 11., 12.};
    auto Fd_host = Kokkos::View<double[num_qps][6], Kokkos::HostSpace>(Fd_data.data());
    auto Fd_mirror = Kokkos::create_mirror(Fd);
    Kokkos::deep_copy(Fd_mirror, Fd_host);
    Kokkos::deep_copy(Fd, Fd_mirror);

    auto FE = Kokkos::View<double[num_nodes][6]>("FE");

    Kokkos::parallel_for("CalculateNodeForces_FE", num_nodes, CalculateNodeForces_FE{first_node, first_qp, num_qps, weight, jacobian, shape_interp, shape_deriv, Fc, Fd, FE});

    auto FE_exact_data = std::array<double, 6>{178., 212., 246., 280., 314., 348.};
    auto FE_exact = Kokkos::View<double[num_nodes][6], Kokkos::HostSpace>(FE_exact_data.data());
    
    auto FE_mirror = Kokkos::create_mirror(FE);
    Kokkos::deep_copy(FE_mirror, FE);
    
    for(int i = 0; i < 6; ++i) {
        EXPECT_EQ(FE_mirror(0, i), FE_exact(0, i));
    }
}

TEST(CalculateNodeForcesTests, FE_TwoNodesTwoQPs) {
    constexpr auto first_node = 0;
    constexpr auto first_qp = 0;
    constexpr auto num_nodes = 2;
    constexpr auto num_qps = 2;

    auto weight = Kokkos::View<double[num_qps]>("weight");
    auto weight_data = std::array<double, 2>{2., 3.};
    auto weight_host = Kokkos::View<double[num_qps], Kokkos::HostSpace>(weight_data.data());
    auto weight_mirror = Kokkos::create_mirror(weight);
    Kokkos::deep_copy(weight_mirror, weight_host);
    Kokkos::deep_copy(weight, weight_mirror);

    auto jacobian = Kokkos::View<double[num_qps]>("jacobian");
    auto jacobian_data = std::array<double, 2>{4., 5.};
    auto jacobian_host = Kokkos::View<double[num_qps], Kokkos::HostSpace>(jacobian_data.data());
    auto jacobian_mirror = Kokkos::create_mirror(jacobian);
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);

    auto shape_interp = Kokkos::View<double[num_nodes][num_qps]>("shape_interp");
    auto shape_interp_data = std::array<double, 4>{6., 7., 8., 9.};
    auto shape_interp_host = Kokkos::View<double[num_nodes][num_qps], Kokkos::HostSpace>(shape_interp_data.data());
    auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);

    auto shape_deriv = Kokkos::View<double[num_nodes][num_qps]>("shape_deriv");
    auto shape_deriv_data = std::array<double, 4>{10., 11., 12., 13.};
    auto shape_deriv_host = Kokkos::View<double[num_nodes][num_qps], Kokkos::HostSpace>(shape_deriv_data.data());
    auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);

    auto Fc = Kokkos::View<double[num_qps][6]>("Fc");
    auto Fc_data = std::array<double, 12>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    auto Fc_host = Kokkos::View<double[num_qps][6], Kokkos::HostSpace>(Fc_data.data());
    auto Fc_mirror = Kokkos::create_mirror(Fc);
    Kokkos::deep_copy(Fc_mirror, Fc_host);
    Kokkos::deep_copy(Fc, Fc_mirror);

    auto Fd = Kokkos::View<double[num_qps][6]>("Fd");
    auto Fd_data = std::array<double, 12>{13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.};
    auto Fd_host = Kokkos::View<double[num_qps][6], Kokkos::HostSpace>(Fd_data.data());
    auto Fd_mirror = Kokkos::create_mirror(Fd);
    Kokkos::deep_copy(Fd_mirror, Fd_host);
    Kokkos::deep_copy(Fd, Fd_mirror);

    auto FE = Kokkos::View<double[num_nodes][6]>("FE");

    Kokkos::parallel_for("CalculateNodeForces_FE", num_nodes, CalculateNodeForces_FE{first_node, first_qp, num_qps, weight, jacobian, shape_interp, shape_deriv, Fc, Fd, FE});

    auto FE_exact_data = std::array<double, 12>{2870., 3076., 3282., 3488., 3694., 3900., 3694., 3956., 4218., 4480., 4742., 5004.};
    auto FE_exact = Kokkos::View<double[num_nodes][6], Kokkos::HostSpace>(FE_exact_data.data());
    
    auto FE_mirror = Kokkos::create_mirror(FE);
    Kokkos::deep_copy(FE_mirror, FE);
    
    for(int i = 0; i < 6; ++i) {
        EXPECT_EQ(FE_mirror(0, i), FE_exact(0, i));
    }

    for(int i = 0; i < 6; ++i) {
        EXPECT_EQ(FE_mirror(1, i), FE_exact(1, i));
    }
}

TEST(CalculateNodeForcesTests, FI_FG_OneNodeOneQP) {
    constexpr auto first_node = 0;
    constexpr auto first_qp = 0;
    constexpr auto num_nodes = 1;
    constexpr auto num_qps = 1;

    auto weight = Kokkos::View<double[num_qps]>("weight");
    auto weight_data = std::array<double, 1>{2.};
    auto weight_host = Kokkos::View<double[num_qps], Kokkos::HostSpace>(weight_data.data());
    auto weight_mirror = Kokkos::create_mirror(weight);
    Kokkos::deep_copy(weight_mirror, weight_host);
    Kokkos::deep_copy(weight, weight_mirror);

    auto jacobian = Kokkos::View<double[num_qps]>("jacobian");
    auto jacobian_data = std::array<double, 1>{3.};
    auto jacobian_host = Kokkos::View<double[num_qps], Kokkos::HostSpace>(jacobian_data.data());
    auto jacobian_mirror = Kokkos::create_mirror(jacobian);
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);

    auto shape_interp = Kokkos::View<double[num_nodes][num_qps]>("shape_interp");
    auto shape_interp_data = std::array<double, 1>{4.};
    auto shape_interp_host = Kokkos::View<double[num_nodes][num_qps], Kokkos::HostSpace>(shape_interp_data.data());
    auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);

    auto shape_deriv = Kokkos::View<double[num_nodes][num_qps]>("shape_deriv");
    auto shape_deriv_data = std::array<double, 1>{5.};
    auto shape_deriv_host = Kokkos::View<double[num_nodes][num_qps], Kokkos::HostSpace>(shape_deriv_data.data());
    auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);

    auto Fig = Kokkos::View<double[num_qps][6]>("Fig");
    auto Fig_data = std::array<double, 6>{1., 2., 3., 4., 5., 6.};
    auto Fig_host = Kokkos::View<double[num_qps][6], Kokkos::HostSpace>(Fig_data.data());
    auto Fig_mirror = Kokkos::create_mirror(Fig);
    Kokkos::deep_copy(Fig_mirror, Fig_host);
    Kokkos::deep_copy(Fig, Fig_mirror);

    auto FIG = Kokkos::View<double[num_nodes][6]>("FIG");

    Kokkos::parallel_for("CalculateNodeForces_FI_FG", num_nodes, CalculateNodeForces_FI_FG{first_node, first_qp, num_qps, weight, jacobian, shape_interp, shape_deriv, Fig, FIG});

    auto FIG_exact_data = std::array<double, 6>{24., 48., 72., 96., 120., 144.};
    auto FIG_exact = Kokkos::View<double[num_nodes][6], Kokkos::HostSpace>(FIG_exact_data.data());
    
    auto FIG_mirror = Kokkos::create_mirror(FIG);
    Kokkos::deep_copy(FIG_mirror, FIG);
    
    for(int i = 0; i < 6; ++i) {
        EXPECT_EQ(FIG_mirror(0, i), FIG_exact(0, i));
    }
}

TEST(CalculateNodeForcesTests, FI_FG_TwoNodesTwoQP) {
    constexpr auto first_node = 0;
    constexpr auto first_qp = 0;
    constexpr auto num_nodes = 2;
    constexpr auto num_qps = 2;

    auto weight = Kokkos::View<double[num_qps]>("weight");
    auto weight_data = std::array<double, 2>{2., 3.};
    auto weight_host = Kokkos::View<double[num_qps], Kokkos::HostSpace>(weight_data.data());
    auto weight_mirror = Kokkos::create_mirror(weight);
    Kokkos::deep_copy(weight_mirror, weight_host);
    Kokkos::deep_copy(weight, weight_mirror);

    auto jacobian = Kokkos::View<double[num_qps]>("jacobian");
    auto jacobian_data = std::array<double, 2>{4., 5.};
    auto jacobian_host = Kokkos::View<double[num_qps], Kokkos::HostSpace>(jacobian_data.data());
    auto jacobian_mirror = Kokkos::create_mirror(jacobian);
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);

    auto shape_interp = Kokkos::View<double[num_nodes][num_qps]>("shape_interp");
    auto shape_interp_data = std::array<double, 4>{6., 7., 8., 9.};
    auto shape_interp_host = Kokkos::View<double[num_nodes][num_qps], Kokkos::HostSpace>(shape_interp_data.data());
    auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);

    auto shape_deriv = Kokkos::View<double[num_nodes][num_qps]>("shape_deriv");
    auto shape_deriv_data = std::array<double, 4>{10., 11., 12., 13.};
    auto shape_deriv_host = Kokkos::View<double[num_nodes][num_qps], Kokkos::HostSpace>(shape_deriv_data.data());
    auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);

    auto Fig = Kokkos::View<double[num_qps][6]>("Fig");
    auto Fig_data = std::array<double, 12>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    auto Fig_host = Kokkos::View<double[num_qps][6], Kokkos::HostSpace>(Fig_data.data());
    auto Fig_mirror = Kokkos::create_mirror(Fig);
    Kokkos::deep_copy(Fig_mirror, Fig_host);
    Kokkos::deep_copy(Fig, Fig_mirror);

    auto FIG = Kokkos::View<double[num_nodes][6]>("FIG");

    Kokkos::parallel_for("CalculateNodeForces_FI_FG", num_nodes, CalculateNodeForces_FI_FG{first_node, first_qp, num_qps, weight, jacobian, shape_interp, shape_deriv, Fig, FIG});

    auto FIG_exact_data = std::array<double, 12>{783., 936., 1089., 1242., 1395., 1548., 1009., 1208., 1407., 1606., 1805., 2004.};
    auto FIG_exact = Kokkos::View<double[num_nodes][6], Kokkos::HostSpace>(FIG_exact_data.data());
    
    auto FIG_mirror = Kokkos::create_mirror(FIG);
    Kokkos::deep_copy(FIG_mirror, FIG);
    
    for(int i = 0; i < 6; ++i) {
        EXPECT_EQ(FIG_mirror(0, i), FIG_exact(0, i));
    }

    for(int i = 0; i < 6; ++i) {
        EXPECT_EQ(FIG_mirror(1, i), FIG_exact(1, i));
    }
}

}