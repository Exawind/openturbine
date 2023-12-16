#pragma once

#include "KokkosBlas.hpp"

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
    template <typename VectorView, typename MatrixView>
    void TangentOperator(VectorView psi, MatrixView tangent_operator) {
        static_assert(VectorView::rank == 1);
        static_assert(MatrixView::rank == 2);
        Kokkos::deep_copy(tangent_operator, 0.);
        auto populate_matrix = KOKKOS_LAMBDA(size_t) {
            tangent_operator(0, 0) = 1.;
            tangent_operator(1, 1) = 1.;
            tangent_operator(2, 2) = 1.;
            tangent_operator(3, 3) = 1.;
            tangent_operator(4, 4) = 1.;
            tangent_operator(5, 5) = 1.;
        };
        Kokkos::parallel_for(1, populate_matrix);
        auto phi = KokkosBlas::nrm2(psi);
        if (phi != 0.) {
            auto psi_tilde = gen_alpha_solver::create_cross_product_matrix(psi);
            auto psi_tilde2 = gen_alpha_solver::create_cross_product_matrix(psi);
            KokkosBlas::gemm("N", "N", 1., psi_tilde, psi_tilde, 0., psi_tilde2);
            auto R =
                Kokkos::subview(tangent_operator, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
            KokkosBlas::axpy((1. - cos(phi)) / (phi * phi), psi_tilde, R);
            KokkosBlas::axpy((1. - sin(phi) / phi) / (phi * phi), psi_tilde2, R);
        }
    }

private:
    Kokkos::View<double*> position_vectors_;
    StiffnessMatrix stiffness_matrix_;
    UserDefinedQuadrature quadrature_;
};

}  // namespace openturbine::gebt_poc