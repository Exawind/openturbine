#pragma once

#include "src/gebt_poc/solver.h"
#include "src/gen_alpha_poc/linearization_parameters.h"
#include "src/gen_alpha_poc/state.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

/*!
 * Calculates the residual vector and iteration matrix for a static beam element
 */
class StaticBeamLinearizationParameters : public gen_alpha_solver::LinearizationParameters {
public:
    static constexpr size_t kNumberOfLieGroupComponents = 7;
    static constexpr size_t kNumberOfLieAlgebraComponents = 6;
    static constexpr size_t kNumberOfConstraints = 3;

    /// Default constructor with a 5 node beam element, 6x6 stiffness matrix, and 7 point
    /// Gauss-Legendre quadrature rule
    StaticBeamLinearizationParameters();

    StaticBeamLinearizationParameters(
        Kokkos::View<double*> position_vectors, StiffnessMatrix stiffness_matrix,
        UserDefinedQuadrature quadrature
    );

    virtual Kokkos::View<double*> ResidualVector(
        const Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
        const Kokkos::View<double*> lagrange_multipliers
    ) override;

    virtual Kokkos::View<double**> IterationMatrix(
        const double& h, const double& beta_prime, const double& gamma_prime,
        const Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
        const Kokkos::View<double*> lagrange_mults
    ) override;

private:
    Kokkos::View<double*> position_vectors_;
    StiffnessMatrix stiffness_matrix_;
    UserDefinedQuadrature quadrature_;
};

}  // namespace openturbine::gebt_poc
