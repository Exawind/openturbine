#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "elements/beams/calculate_jacobian.hpp"
#include "elements/beams/interpolation.hpp"

namespace openturbine::tests {

TEST(CalculateJacobian, LinearElement) {
    constexpr size_t num_elems{1};
    constexpr size_t num_nodes{2};
    constexpr size_t num_qps{1};

    const auto num_nodes_per_elem = Kokkos::View<size_t*>("num_nodes", num_elems);
    const auto num_qps_per_elem = Kokkos::View<size_t*>("num_qps", num_elems);
    const auto shape_derivative =
        Kokkos::View<double***>("shape_derivative", num_elems, num_nodes, num_qps);
    const auto node_position_rotation =
        Kokkos::View<double** [7]>("node_position_rotation", num_elems, num_nodes);
    const auto qp_position_derivative =
        Kokkos::View<double** [3]>("position_derivative", num_elems, num_qps);
    const auto qp_jacobian = Kokkos::View<double**>("jacobian", num_elems, num_qps);

    const auto host_num_nodes = Kokkos::create_mirror_view(num_nodes_per_elem);
    const auto host_num_qps = Kokkos::create_mirror_view(num_qps_per_elem);
    const auto host_shape_derivative = Kokkos::create_mirror_view(shape_derivative);
    const auto host_node_position_rotation = Kokkos::create_mirror_view(node_position_rotation);
    const auto host_qp_position_derivative = Kokkos::create_mirror_view(qp_position_derivative);
    const auto host_qp_jacobian = Kokkos::create_mirror_view(qp_jacobian);

    // Set values for a linear element from (-1.,0.,0.) -> (1.,0.,0.)
    host_node_position_rotation(0, 0, 0) = -1.;  // node 1, x
    host_node_position_rotation(0, 0, 1) = 0.;   // node 1, y
    host_node_position_rotation(0, 0, 2) = 0.;   // node 1, z
    host_node_position_rotation(0, 1, 0) = 1.;   // node 2, x
    host_node_position_rotation(0, 1, 1) = 0.;   // node 2, y
    host_node_position_rotation(0, 1, 2) = 0.;   // node 2, z

    host_num_nodes(0) = num_nodes;
    host_num_qps(0) = num_qps;

    // Shape function derivatives for a linear element
    host_shape_derivative(0, 0, 0) = -0.5;  // dN1/dξ at ξ = 0
    host_shape_derivative(0, 1, 0) = 0.5;   // dN2/dξ at ξ = 0

    Kokkos::deep_copy(num_nodes_per_elem, host_num_nodes);
    Kokkos::deep_copy(num_qps_per_elem, host_num_qps);
    Kokkos::deep_copy(shape_derivative, host_shape_derivative);
    Kokkos::deep_copy(node_position_rotation, host_node_position_rotation);

    const auto calculate_jacobian = CalculateJacobian{num_nodes_per_elem,     num_qps_per_elem,
                                                      shape_derivative,       node_position_rotation,
                                                      qp_position_derivative, qp_jacobian};
    Kokkos::parallel_for("calculate_jacobian", 1, calculate_jacobian);

    auto host_jacobian = Kokkos::create_mirror_view(qp_jacobian);
    Kokkos::deep_copy(host_jacobian, qp_jacobian);

    // For a linear element of length 2., the Jacobian should be 1.
    ASSERT_EQ(host_jacobian.extent(0), num_elems);  // 1 element
    ASSERT_EQ(host_jacobian.extent(1), num_qps);    // 1 quadrature point
    EXPECT_DOUBLE_EQ(host_jacobian(0, 0), 1.);

    auto host_position_derivative = Kokkos::create_mirror_view(qp_position_derivative);
    Kokkos::deep_copy(host_position_derivative, qp_position_derivative);

    // Should be a unit vector in x direction
    ASSERT_EQ(host_position_derivative.extent(0), num_elems);  // 1 element
    ASSERT_EQ(host_position_derivative.extent(1), num_qps);    // 1 quadrature point
    ASSERT_EQ(host_position_derivative.extent(2), 3);          // 3 dimensions
    EXPECT_DOUBLE_EQ(host_position_derivative(0, 0, 0), 1.);
    EXPECT_DOUBLE_EQ(host_position_derivative(0, 0, 1), 0.);
    EXPECT_DOUBLE_EQ(host_position_derivative(0, 0, 2), 0.);
}

TEST(CalculateJacobian, FourthOrderElement) {
    constexpr size_t num_elems{1};
    constexpr size_t num_nodes{5};  // 4th order = 5 nodes
    constexpr size_t num_qps{5};    // 5 quadrature points

    const auto num_nodes_per_elem = Kokkos::View<size_t*>("num_nodes", num_elems);
    const auto num_qps_per_elem = Kokkos::View<size_t*>("num_qps", num_elems);
    const auto shape_derivative =
        Kokkos::View<double***>("shape_derivative", num_elems, num_nodes, num_qps);
    const auto node_position_rotation =
        Kokkos::View<double** [7]>("node_position_rotation", num_elems, num_nodes);
    const auto qp_position_derivative =
        Kokkos::View<double** [3]>("position_derivative", num_elems, num_qps);
    const auto qp_jacobian = Kokkos::View<double**>("jacobian", num_elems, num_qps);

    const auto host_num_nodes = Kokkos::create_mirror_view(num_nodes_per_elem);
    const auto host_num_qps = Kokkos::create_mirror_view(num_qps_per_elem);
    const auto host_shape_derivative = Kokkos::create_mirror_view(shape_derivative);
    const auto host_node_position_rotation = Kokkos::create_mirror_view(node_position_rotation);
    const auto host_qp_position_derivative = Kokkos::create_mirror_view(qp_position_derivative);
    const auto host_qp_jacobian = Kokkos::create_mirror_view(qp_jacobian);

    // Node positions
    // Node 1: 0., 0., 0.
    host_node_position_rotation(0, 0, 0) = 0.;
    host_node_position_rotation(0, 0, 1) = 0.;
    host_node_position_rotation(0, 0, 2) = 0.;
    // Node 2: 0.16237631096713473, 0.17578464768961147, 0.1481911137890286
    host_node_position_rotation(0, 1, 0) = 0.16237631096713473;
    host_node_position_rotation(0, 1, 1) = 0.17578464768961147;
    host_node_position_rotation(0, 1, 2) = 0.1481911137890286;
    // Node 3: 0.25, 1., 1.1875
    host_node_position_rotation(0, 2, 0) = 0.25;
    host_node_position_rotation(0, 2, 1) = 1.0;
    host_node_position_rotation(0, 2, 2) = 1.1875;
    // Node 4: -0.30523345382427747, 2.4670724951675314, 2.953849702537502
    host_node_position_rotation(0, 3, 0) = -0.30523345382427747;
    host_node_position_rotation(0, 3, 1) = 2.4670724951675314;
    host_node_position_rotation(0, 3, 2) = 2.953849702537502;
    // Node 5: -1., 3.5, 4.
    host_node_position_rotation(0, 4, 0) = -1.0;
    host_node_position_rotation(0, 4, 1) = 3.5;
    host_node_position_rotation(0, 4, 2) = 4.0;

    // For a 4th order element, following are the shape function derivatives
    const auto nodes = GenerateGLLPoints(4);
    constexpr std::array<double, 5> qp_locations = {
        // Gauss quadrature points
        -0.9061798459386640, -0.5384693101056831, 0., 0.5384693101056831, 0.9061798459386640
    };
    for (size_t qp = 0; qp < num_qps; ++qp) {
        std::vector<double> weights;
        LagrangePolynomialDerivWeights(qp_locations[qp], nodes, weights);
        for (size_t node = 0; node < nodes.size(); ++node) {
            host_shape_derivative(0, node, qp) = weights[node];
        }
    }

    host_num_nodes(0) = num_nodes;
    host_num_qps(0) = num_qps;

    Kokkos::deep_copy(num_nodes_per_elem, host_num_nodes);
    Kokkos::deep_copy(num_qps_per_elem, host_num_qps);
    Kokkos::deep_copy(shape_derivative, host_shape_derivative);
    Kokkos::deep_copy(node_position_rotation, host_node_position_rotation);

    const auto calculate_jacobian = CalculateJacobian{num_nodes_per_elem,     num_qps_per_elem,
                                                      shape_derivative,       node_position_rotation,
                                                      qp_position_derivative, qp_jacobian};
    Kokkos::parallel_for("calculate_jacobian", 1, calculate_jacobian);
    Kokkos::deep_copy(host_qp_jacobian, qp_jacobian);

    // Expected jacobians at each quadrature point (from BeamDyn)
    ASSERT_EQ(host_qp_jacobian.extent(0), num_elems);  // 1 element
    ASSERT_EQ(host_qp_jacobian.extent(1), num_qps);    // 5 quadrature points

    constexpr std::array<double, 5> expected_jacobians = {
        0.671587005850145, 1.509599209717606, 2.861380785564898, 4.097191592895187, 4.880926263217582
    };
    for (size_t qp = 0; qp < num_qps; ++qp) {
        EXPECT_NEAR(host_qp_jacobian(0, qp), expected_jacobians[qp], 1e-12);
    }

    // Verify that position derivatives are unit vectors
    ASSERT_EQ(host_qp_position_derivative.extent(0), num_elems);  // 1 element
    ASSERT_EQ(host_qp_position_derivative.extent(1), num_qps);    // 5 quadrature points
    ASSERT_EQ(host_qp_position_derivative.extent(2), 3);          // 3 dimensions

    Kokkos::deep_copy(host_qp_position_derivative, qp_position_derivative);
    for (size_t qp = 0; qp < num_qps; ++qp) {
        const auto magnitude = std::sqrt(
            host_qp_position_derivative(0, qp, 0) * host_qp_position_derivative(0, qp, 0) +
            host_qp_position_derivative(0, qp, 1) * host_qp_position_derivative(0, qp, 1) +
            host_qp_position_derivative(0, qp, 2) * host_qp_position_derivative(0, qp, 2)
        );
        EXPECT_NEAR(magnitude, 1., 1e-12);  // unit vector
    }
}

}  // namespace openturbine::tests
