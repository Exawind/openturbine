#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "constraints/calculate_revolute_joint_constraint.hpp"

namespace openturbine::tests {

struct ExecuteCalculateRevoluteJointConstraint {
    int i_constraint;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3]>::const_type X0;
    Kokkos::View<double* [3][3]>::const_type axes;
    Kokkos::View<double* [7]>::const_type constraint_inputs;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double* [6]> residual_terms;
    Kokkos::View<double* [6][6]> base_gradient_terms;
    Kokkos::View<double* [6][6]> target_gradient_terms;

    KOKKOS_FUNCTION
    void operator()(int) const {
        CalculateRevoluteJointConstraint{i_constraint,
                                         base_node_index,
                                         target_node_index,
                                         X0,
                                         axes,
                                         constraint_inputs,
                                         node_u,
                                         residual_terms,
                                         base_gradient_terms,
                                         target_gradient_terms}();
    }
};

TEST(CalculateRevoluteJointConstraintTests, OneConstraint) {
    const auto target_node_index = Kokkos::View<size_t[1]>("target_node_index");
    constexpr auto target_node_index_host_data = std::array<size_t, 1>{1UL};
    const auto target_node_index_host =
        Kokkos::View<size_t[1], Kokkos::HostSpace>::const_type(target_node_index_host_data.data());
    Kokkos::deep_copy(target_node_index, target_node_index_host);

    const auto base_node_index = Kokkos::View<size_t[1]>("base_node_index");
    constexpr auto base_node_index_host_data = std::array<size_t, 1>{2UL};
    const auto base_node_index_host =
        Kokkos::View<size_t[1], Kokkos::HostSpace>::const_type(base_node_index_host_data.data());
    Kokkos::deep_copy(base_node_index, base_node_index_host);

    const auto X0 = Kokkos::View<double[1][3]>("X0");
    constexpr auto X0_host_data = std::array{1., 2., 3.};
    const auto X0_host =
        Kokkos::View<double[1][3], Kokkos::HostSpace>::const_type(X0_host_data.data());
    const auto X0_mirror = Kokkos::create_mirror(X0);
    Kokkos::deep_copy(X0_mirror, X0_host);
    Kokkos::deep_copy(X0, X0_mirror);

    const auto axes = Kokkos::View<double[1][3][3]>("axes");
    constexpr auto axes_host_data = std::array{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    const auto axes_host =
        Kokkos::View<double[1][3][3], Kokkos::HostSpace>::const_type(axes_host_data.data());
    const auto axes_mirror = Kokkos::create_mirror(axes);
    Kokkos::deep_copy(axes_mirror, axes_host);
    Kokkos::deep_copy(axes, axes_mirror);

    const auto constraint_inputs = Kokkos::View<double[1][7]>("constraint_inputs");
    constexpr auto constraint_inputs_host_data = std::array{4., 5., 6., 7., 8., 9., 10.};
    const auto constraint_inputs_host =
        Kokkos::View<double[1][7], Kokkos::HostSpace>::const_type(constraint_inputs_host_data.data()
        );
    const auto constraint_inputs_mirror = Kokkos::create_mirror(constraint_inputs);
    Kokkos::deep_copy(constraint_inputs_mirror, constraint_inputs_host);
    Kokkos::deep_copy(constraint_inputs, constraint_inputs_mirror);

    const auto node_u = Kokkos::View<double[3][7]>("node_u");
    constexpr auto node_u_host_data =
        std::array{0.,  0.,  0.,  0.,  0.,  0.,  0.,  11., 12., 13., 14.,
                   15., 16., 17., 18., 19., 20., 21., 22., 23., 24.};
    const auto node_u_host =
        Kokkos::View<double[3][7], Kokkos::HostSpace>::const_type(node_u_host_data.data());
    const auto node_u_mirror = Kokkos::create_mirror(node_u);
    Kokkos::deep_copy(node_u_mirror, node_u_host);
    Kokkos::deep_copy(node_u, node_u_mirror);

    const auto residual_terms = Kokkos::View<double[1][6]>("residual_terms");
    const auto base_gradient_terms = Kokkos::View<double[1][6][6]>("base_gradient_terms");
    const auto target_gradient_terms = Kokkos::View<double[1][6][6]>("target_gradient_terms");

    Kokkos::parallel_for(
        "CalculateRevoluteJointConstraint", 1,
        ExecuteCalculateRevoluteJointConstraint{
            0, base_node_index, target_node_index, X0, axes, constraint_inputs, node_u,
            residual_terms, base_gradient_terms, target_gradient_terms
        }
    );

    const auto residual_terms_mirror = Kokkos::create_mirror(residual_terms);
    Kokkos::deep_copy(residual_terms_mirror, residual_terms);

    constexpr auto residual_terms_exact_data =
        std::array{-5900., -2385., -4162., 97314000., 62379744., 0.};
    const auto residual_terms_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(residual_terms_exact_data.data());

    for (auto i = 0U; i < 6U; ++i) {
        EXPECT_NEAR(residual_terms_mirror(0, i), residual_terms_exact(0, i), 1.e-15);
    }

    const auto base_gradient_terms_mirror = Kokkos::create_mirror(base_gradient_terms);
    Kokkos::deep_copy(base_gradient_terms_mirror, base_gradient_terms);

    constexpr auto base_gradient_terms_exact_data = std::array{
        -1., 0.,  0.,  0.,         0.,         0.,         // Row 1
        0.,  -1., 0.,  0.,         0.,         0.,         // Row 2
        0.,  0.,  -1., 0.,         0.,         0.,         // Row 3
        0.,  0.,  0.,  -10930136., -15850520., 24566248.,  // Row 4
        0.,  0.,  0.,  -5585804.,  -8091272.,  12549292.,  // Row 5
        0.,  0.,  0.,  0.,         0.,         0.          // Row 6
    };
    const auto base_gradient_terms_exact =
        Kokkos::View<double[1][6][6], Kokkos::HostSpace>::const_type(
            base_gradient_terms_exact_data.data()
        );

    for (auto i = 0U; i < 6U; ++i) {
        for (auto j = 0U; j < 6U; ++j) {
            EXPECT_NEAR(
                base_gradient_terms_mirror(0, i, j), base_gradient_terms_exact(0, i, j), 1.e-15
            );
        }
    }

    const auto target_gradient_terms_mirror = Kokkos::create_mirror(target_gradient_terms);
    Kokkos::deep_copy(target_gradient_terms_mirror, target_gradient_terms);

    for (auto i = 0U; i < 6U; ++i) {
        for (auto j = 0U; j < 6U; ++j) {
            EXPECT_NEAR(
                target_gradient_terms_mirror(0, i, j), -base_gradient_terms_exact(0, i, j), 1.e-15
            );
        }
    }
}

}  // namespace openturbine::tests
