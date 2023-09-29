#include "src/gebt_poc/element.h"

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

    auto two_norm = [](const std::array<double, 3>& vector) {
        return std::sqrt(std::pow(vector[0], 2) + std::pow(vector[1], 2) + std::pow(vector[2], 2));
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

    auto jacobian = 0.;
    auto gup0 = std::array<double, 3>{0., 0., 0.};
    for (size_t i = 0; i < nodes.size(); ++i) {
        gup0[0] = dot_product(shape_derivatives, x_components);
        gup0[1] = dot_product(shape_derivatives, y_components);
        gup0[2] = dot_product(shape_derivatives, z_components);
        jacobian = two_norm(gup0);
    }
    return jacobian;
}

}  // namespace openturbine::gebt_poc
