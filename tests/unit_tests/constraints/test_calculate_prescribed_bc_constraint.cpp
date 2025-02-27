#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "constraints/calculate_prescribed_bc_constraint.hpp"

namespace openturbine::tests {

struct ExecuteCalculatePrescribedBCConstraint {
    Kokkos::View<double[3]>::const_type X0;
    Kokkos::View<double[7]>::const_type inputs;
    Kokkos::View<double[7]>::const_type node_u;
    Kokkos::View<double[6]> residual_terms;
    Kokkos::View<double[6][6]> target_gradient_terms;

    KOKKOS_FUNCTION
    void operator()(int) const {
        CalculatePrescribedBCConstraint(X0, inputs, node_u, residual_terms, target_gradient_terms);
    }
};

TEST(CalculatePrescribedBCConstraintTests, OneConstraint) {
    const auto X0 = Kokkos::View<double[3]>("X0");
    constexpr auto X0_host_data = std::array{1., 2., 3.};
    const auto X0_host = Kokkos::View<double[3], Kokkos::HostSpace>::const_type(X0_host_data.data());
    const auto X0_mirror = Kokkos::create_mirror(X0);
    Kokkos::deep_copy(X0_mirror, X0_host);
    Kokkos::deep_copy(X0, X0_mirror);

    const auto constraint_inputs = Kokkos::View<double[7]>("constraint_inputs");
    constexpr auto constraint_inputs_host_data = std::array{4., 5., 6., 7., 8., 9., 10.};
    const auto constraint_inputs_host =
        Kokkos::View<double[7], Kokkos::HostSpace>::const_type(constraint_inputs_host_data.data());
    const auto constraint_inputs_mirror = Kokkos::create_mirror(constraint_inputs);
    Kokkos::deep_copy(constraint_inputs_mirror, constraint_inputs_host);
    Kokkos::deep_copy(constraint_inputs, constraint_inputs_mirror);

    const auto node_u = Kokkos::View<double[7]>("node_u");
    constexpr auto node_u_host_data = std::array{11., 12., 13., 14., 15., 16., 17.};
    const auto node_u_host =
        Kokkos::View<double[7], Kokkos::HostSpace>::const_type(node_u_host_data.data());
    const auto node_u_mirror = Kokkos::create_mirror(node_u);
    Kokkos::deep_copy(node_u_mirror, node_u_host);
    Kokkos::deep_copy(node_u, node_u_mirror);

    const auto residual_terms = Kokkos::View<double[6]>("residual_terms");
    const auto target_gradient_terms = Kokkos::View<double[6][6]>("target_gradient_terms");

    Kokkos::parallel_for(
        "CalculatePrescribedBCConstraint", 1,
        ExecuteCalculatePrescribedBCConstraint{
            X0, constraint_inputs, node_u, residual_terms, target_gradient_terms}
    );

    const auto residual_terms_mirror = Kokkos::create_mirror(residual_terms);
    Kokkos::deep_copy(residual_terms_mirror, residual_terms);

    constexpr auto residual_terms_exact_data = std::array{
        -790., -411., -620., -50.666666666666657, -7.1054273576010019e-15, -101.33333333333343
    };
    const auto residual_terms_exact =
        Kokkos::View<double[6], Kokkos::HostSpace>::const_type(residual_terms_exact_data.data());

    for (auto i = 0U; i < 6U; ++i) {
        EXPECT_NEAR(residual_terms_mirror(i), residual_terms_exact(i), 1.e-12);
    }

    const auto target_gradient_terms_mirror = Kokkos::create_mirror(target_gradient_terms);
    Kokkos::deep_copy(target_gradient_terms_mirror, target_gradient_terms);

    // clang-format off
    constexpr auto target_gradient_terms_exact_data = std::array{
        1., 0., 0., 0., 0., 0.,  // Row 1
        0., 1., 0., 0., 0., 0.,  // Row 2
        0., 0., 1., 0., 0., 0.,  // Row 3
        0., 0., 0., 961.99999999999977, 50.666666666666714, -1.3333333333333379,  // Row 4
        0., 0., 0., -50.666666666666714, 962.6666666666664, 25.333333333333329,   // Row 5
        0., 0., 0., -1.3333333333333308, -25.333333333333329, 959.99999999999977  // Row 6
    };
    // clang-format on
    const auto target_gradient_terms_exact =
        Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type(
            target_gradient_terms_exact_data.data()
        );

    for (auto i = 0U; i < 6U; ++i) {
        for (auto j = 0U; j < 6U; ++j) {
            EXPECT_NEAR(
                target_gradient_terms_mirror(i, j), target_gradient_terms_exact(i, j), 1.e-12
            );
        }
    }
}

}  // namespace openturbine::tests
