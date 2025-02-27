#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "constraints/calculate_revolute_joint_force.hpp"

namespace openturbine::tests {

struct ExecuteCalculateRevoluteJointForce {
    Kokkos::View<double[3][3]>::const_type axes;
    Kokkos::View<double[7]>::const_type constraint_inputs;
    Kokkos::View<double[7]>::const_type node_u;
    Kokkos::View<double[6]> residual_terms;

    KOKKOS_FUNCTION
    void operator()(int) const {
        CalculateRevoluteJointForce(axes, constraint_inputs,
                                    node_u,       residual_terms);
    }
};

TEST(CalculateRevoluteJointForceTests, OneConstraint) {
    const auto axes = Kokkos::View<double[3][3]>("axes");
    constexpr auto axes_host_data = std::array{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    const auto axes_host =
        Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type(axes_host_data.data());
    const auto axes_mirror = Kokkos::create_mirror(axes);
    Kokkos::deep_copy(axes_mirror, axes_host);
    Kokkos::deep_copy(axes, axes_mirror);

    const auto constraint_inputs = Kokkos::View<double[7]>("constraint_inputs");
    constexpr auto constraint_inputs_host_data = std::array{4., 5., 6., 7., 8., 9., 10.};
    const auto constraint_inputs_host =
        Kokkos::View<double[7], Kokkos::HostSpace>::const_type(constraint_inputs_host_data.data()
        );
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

    Kokkos::parallel_for(
        "CalculateRevoluteJointConstraint", 1,
        ExecuteCalculateRevoluteJointForce{axes, constraint_inputs, node_u, residual_terms}
    );

    const auto residual_terms_mirror = Kokkos::create_mirror(residual_terms);
    Kokkos::deep_copy(residual_terms_mirror, residual_terms);

    constexpr auto residual_terms_exact_data = std::array{0., 0., 0., 11032., 4816., 8008.};
    const auto residual_terms_exact =
        Kokkos::View<double[6], Kokkos::HostSpace>::const_type(residual_terms_exact_data.data());

    for (auto i = 0U; i < 6U; ++i) {
        EXPECT_NEAR(residual_terms_mirror(i), residual_terms_exact(i), 1.e-15);
    }
}

}  // namespace openturbine::tests
