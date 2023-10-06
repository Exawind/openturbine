#pragma once

#include "src/gebt_poc/element.h"
#include "src/gebt_poc/section.h"

namespace openturbine::gebt_poc {

constexpr size_t kNumberOfLieAlgebraComponents = 7;
constexpr size_t kNumberOfLieGroupComponents = 6;
constexpr size_t kNumberOfVectorComponents = 3;

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
    );

    // Returns the number of quadrature points
    virtual size_t GetNumberOfQuadraturePoints() const override { return quadrature_points_.size(); }

    // Returns the quadrature points (read only)
    virtual const std::vector<double>& GetQuadraturePoints() const override {
        return quadrature_points_;
    }

    // Returns the quadrature weights (read only)
    virtual const std::vector<double>& GetQuadratureWeights() const override {
        return quadrature_weights_;
    }

private:
    std::vector<double> quadrature_points_;
    std::vector<double> quadrature_weights_;
};

/// Calculates the interpolated values for a nodal quantity (e.g. displacement or position vector) at
/// a given quadrature point
Kokkos::View<double*> Interpolate(Kokkos::View<double*> nodal_values, double quadrature_pt);

/// Calculates the static residual vector for a beam element
Kokkos::View<double*> CalculateStaticResidual(
    const Kokkos::View<double*> position_vectors, const Kokkos::View<double*> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature
);

}  // namespace openturbine::gebt_poc
