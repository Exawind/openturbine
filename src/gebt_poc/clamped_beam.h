#pragma once

#include "src/gebt_poc/linearization_parameters.h"
#include "src/gebt_poc/solver.h"
#include "src/gen_alpha_poc/state.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

/// Calculates the constraint gradient matrix for the clamped beam problem
void BMatrix(Kokkos::View<double**> constraints_gradient_matrix);

/*!
 * Calculates the residual vector and iteration matrix for a static beam element
 */
class ClampedBeamLinearizationParameters : public LinearizationParameters {
public:
    static constexpr size_t kNumberOfLieGroupComponents = 7;
    static constexpr size_t kNumberOfLieAlgebraComponents = 6;
    static constexpr size_t kNumberOfVectorComponents = 3;
    static constexpr double kTolerance = 1e-16;

    /// Define a static beam element with the given position vector for the nodes, 6x6
    /// stiffness matrix, and a quadrature rule
    ClampedBeamLinearizationParameters(
        Kokkos::View<double*> position_vectors, StiffnessMatrix stiffness_matrix,
        UserDefinedQuadrature quadrature
    );

    virtual void ResidualVector(
        Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
        Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double*> residual_vector
    ) override;

    virtual void IterationMatrix(
        const double& h, const double& beta_prime, const double& gamma_prime,
        Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
        Kokkos::View<double*> lagrange_multipliers, Kokkos::View<double**> iteration_matrix
    ) override;

    /// Tangent operator for a single node of the static beam element
    void TangentOperator(
        Kokkos::View<double[kNumberOfVectorComponents]> psi, Kokkos::View<double**> tangent_operator
    );

private:
    Kokkos::View<double*> position_vectors_;
    StiffnessMatrix stiffness_matrix_;
    UserDefinedQuadrature quadrature_;
};

}  // namespace openturbine::gebt_poc
