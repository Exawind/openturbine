#pragma once

#include "src/gebt_poc/element.h"

namespace openturbine::gebt_poc {

/// An abstract class for providing common interface to numerical quadrature rules
class Quadrature {
public:
    virtual ~Quadrature() = default;

    /// Returns the number of quadrature points
    virtual size_t GetNumberOfQuadraturePoints() const = 0;

    /// Returns the quadrature points (read only)
    virtual const std::vector<double>& GetQuadraturePoints() const = 0;

    /// Returns the quadrature weights (read only)
    virtual const std::vector<double>& GetQuadratureWeights() const = 0;
};

/// A concrete quadrature rule where the quadrature points and weights are provided by the user
class UserDefinedQuadrature : public Quadrature {
public:
    UserDefinedQuadrature(
        std::vector<double> quadrature_points, std::vector<double> quadrature_weights
    )
        : quadrature_points_(std::move(quadrature_points)),
          quadrature_weights_(std::move(quadrature_weights)) {}

    /// Returns the number of quadrature points
    virtual size_t GetNumberOfQuadraturePoints() const override { return quadrature_points_.size(); }

    /// Returns the quadrature points (read only)
    virtual const std::vector<double>& GetQuadraturePoints() const override {
        return quadrature_points_;
    }

    /// Returns the quadrature weights (read only)
    virtual const std::vector<double>& GetQuadratureWeights() const override {
        return quadrature_weights_;
    }

private:
    std::vector<double> quadrature_points_;
    std::vector<double> quadrature_weights_;
};

}  // namespace openturbine::gebt_poc
