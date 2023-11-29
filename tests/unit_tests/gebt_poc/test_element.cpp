#include <gtest/gtest.h>
#include <initializer_list>

#include "src/gebt_poc/element.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

auto MakePointVector(std::initializer_list<Point> point_list ) {
  auto points = Kokkos::View<double*[3]>("points", point_list.size());
  auto host_points = Kokkos::create_mirror(points);
  std::size_t i = 0;
  for(auto& point : point_list) {
    host_points(i, 0) = point.GetXComponent();
    host_points(i, 1) = point.GetYComponent();
    host_points(i, 2) = point.GetZComponent();
    ++i;
  }
  Kokkos::deep_copy(points, host_points);
  return points;
}

TEST(ElementTest, JacobianFor1DLinearElement) {
    auto shape_derivatives = gen_alpha_solver::create_vector(LagrangePolynomialDerivative(1, -1.));

    // Nodes are located at GLL points i.e. (-1, 0., 0.) and (1., 0., 0.)
    auto node_1 = Point(-1., 0., 0.);
    auto node_2 = Point(1., 0., 0.);
    auto nodes_1 = MakePointVector({node_1, node_2});
    auto jacobian_1 = CalculateJacobian(nodes_1, shape_derivatives);

    // ** For 1-D elements, the Jacobian is always length of the element / 2 **
    // Length of the element is 2, so Jacobian is 1
    EXPECT_DOUBLE_EQ(jacobian_1, 1.);

    auto node_3 = Point(-2., 0., 0.);
    auto node_4 = Point(2., 0., 0.);
    auto nodes_2 = MakePointVector({node_3, node_4});
    auto jacobian_2 = CalculateJacobian(nodes_2, shape_derivatives);

    // Length of the element is 4, so Jacobian is 2
    EXPECT_DOUBLE_EQ(jacobian_2, 2.);

    auto node_5 = Point(0., 0., 0.);
    auto node_6 = Point(1., 0., 0.);
    auto nodes_3 = MakePointVector({node_5, node_6});
    auto jacobian_3 = CalculateJacobian(nodes_3, shape_derivatives);

    // Length of the element is 1, so Jacobian is 0.5
    EXPECT_DOUBLE_EQ(jacobian_3, 0.5);
}

TEST(ElementTest, JacobianFor1DFourthOrderElement) {
    auto shape_derivatives_1 = gen_alpha_solver::create_vector(LagrangePolynomialDerivative(4, -0.9061798459386640));
    auto shape_derivatives_2 = gen_alpha_solver::create_vector(LagrangePolynomialDerivative(4, -0.5384693101056831));
    auto shape_derivatives_3 = gen_alpha_solver::create_vector(LagrangePolynomialDerivative(4, 0.));
    auto shape_derivatives_4 = gen_alpha_solver::create_vector(LagrangePolynomialDerivative(4, 0.5384693101056831));
    auto shape_derivatives_5 = gen_alpha_solver::create_vector(LagrangePolynomialDerivative(4, 0.9061798459386640));

    auto node_1 = Point(0.0, 0.0, 0.0);
    auto node_2 = Point(0.16237631096713473, 0.17578464768961147, 0.1481911137890286);
    auto node_3 = Point(0.25, 1., 1.1875);
    auto node_4 = Point(-0.30523345382427747, 2.4670724951675314, 2.953849702537502);
    auto node_5 = Point(-1., 3.5, 4.);
    auto nodes = MakePointVector({node_1, node_2, node_3, node_4, node_5});

    auto jacobian_1 = CalculateJacobian(nodes, shape_derivatives_1);
    auto jacobian_2 = CalculateJacobian(nodes, shape_derivatives_2);
    auto jacobian_3 = CalculateJacobian(nodes, shape_derivatives_3);
    auto jacobian_4 = CalculateJacobian(nodes, shape_derivatives_4);
    auto jacobian_5 = CalculateJacobian(nodes, shape_derivatives_5);

    EXPECT_DOUBLE_EQ(jacobian_1, 0.671587005850145);
    EXPECT_DOUBLE_EQ(jacobian_2, 1.509599209717606);
    EXPECT_DOUBLE_EQ(jacobian_3, 2.861380785564898);
    EXPECT_DOUBLE_EQ(jacobian_4, 4.097191592895187);
    EXPECT_DOUBLE_EQ(jacobian_5, 4.880926263217582);
}

}  // namespace openturbine::gebt_poc::tests
