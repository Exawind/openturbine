#pragma once

#include "src/gebt_poc/element.h"

namespace openturbine::gebt_poc {

/// An abstract for providing common interface for numerical quadrature rules
class QuadratureRule {
public:
    virtual ~QuadratureRule() = default;

    /// Returns the number of quadrature points
    virtual size_t GetNumQuadraturePoints() const = 0;

    /// Returns the quadrature points
    virtual std::vector<double> GetQuadraturePoints() const = 0;

    /// Returns the quadrature weights
    virtual std::vector<double> GetQuadratureWeights() const = 0;
};

/// A concrete quadrature rule where the quadrature points and weights are provided by the user
class UserDefinedQuadratureRule : public QuadratureRule {
public:
    UserDefinedQuadratureRule(
        std::vector<double> quadrature_points, std::vector<double> quadrature_weights
    );

    // Returns the number of quadrature points
    virtual size_t GetNumQuadraturePoints() const override { return quadrature_points_.size(); }

    // Returns the quadrature points
    virtual std::vector<double> GetQuadraturePoints() const override { return quadrature_points_; }

    // Returns the quadrature weights
    virtual std::vector<double> GetQuadratureWeights() const override { return quadrature_weights_; }

private:
    std::vector<double> quadrature_points_;   //< Quadrature points
    std::vector<double> quadrature_weights_;  //< Quadrature weights
};

}  // namespace openturbine::gebt_poc
