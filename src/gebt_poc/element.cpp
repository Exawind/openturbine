#include "src/gebt_poc/element.h"

#include <KokkosBlas.hpp>

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

double CalculateJacobian(
    const std::vector<Point>& nodes, const std::vector<double>& shape_derivatives
) {
    auto dot_product = [](const std::vector<double>& vector_1, const std::vector<double>& vector_2) {
        auto result = 0.;
        for (size_t i = 0; i < vector_1.size(); ++i) {
            result += vector_1[i] * vector_2[i];
        }
        return result;
    };

    // Get all the x, y, z components of the nodes in three separate vectors
    std::vector<double> x_components{};
    std::vector<double> y_components{};
    std::vector<double> z_components{};
    for (const auto& node : nodes) {
        x_components.push_back(node.GetXComponent());
        y_components.push_back(node.GetYComponent());
        z_components.push_back(node.GetZComponent());
    }

    auto v = gen_alpha_solver::Vector();
    auto jacobian = 0.;
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto v0 = dot_product(shape_derivatives, x_components);
        auto v1 = dot_product(shape_derivatives, y_components);
        auto v2 = dot_product(shape_derivatives, z_components);

        v = gen_alpha_solver::Vector(v0, v1, v2);
        jacobian = v.Length();
    }
    return jacobian;
}

double CalculateJacobian(VectorFieldView::const_type nodes, View1D::const_type shape_derivatives) {
    auto v0 = KokkosBlas::dot(shape_derivatives, Kokkos::subview(nodes, Kokkos::ALL, 0));
    auto v1 = KokkosBlas::dot(shape_derivatives, Kokkos::subview(nodes, Kokkos::ALL, 1));
    auto v2 = KokkosBlas::dot(shape_derivatives, Kokkos::subview(nodes, Kokkos::ALL, 2));

    return std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
}

}  // namespace openturbine::gebt_poc
