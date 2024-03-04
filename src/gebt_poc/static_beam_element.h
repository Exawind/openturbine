#pragma once

#include <KokkosBlas.hpp>

#include "src/gebt_poc/force.h"
#include "src/gebt_poc/linearization_parameters.h"
#include "src/gebt_poc/solver.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/state.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

void Convert2DViewTo1DView(View2D::const_type view, View1D result);

/// Calculates the constraint gradient matrix for the clamped beam problem
void BMatrix(View2D constraints_gradient_matrix);

/*!
 * Calculates the residual vector and iteration matrix for a static beam element
 */
class StaticBeamLinearizationParameters : public LinearizationParameters {
public:
    /// Define a static beam element with the given position vector for the nodes, 6x6
    /// stiffness matrix, and a quadrature rule
    StaticBeamLinearizationParameters(
        View1D position_vectors, StiffnessMatrix stiffness_matrix, UserDefinedQuadrature quadrature
    );

    virtual void ResidualVector(
        LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type velocity,
        LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_multipliers,
        const gen_alpha_solver::TimeStepper& time_stepper, View1D residual_vector
    ) override;

    virtual void IterationMatrix(
        double h, double beta_prime, double gamma_prime, LieGroupFieldView::const_type gen_coords,
        LieAlgebraFieldView::const_type delta_gen_coords, LieAlgebraFieldView::const_type velocity,
        LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_multipliers,
        View2D iteration_matrix
    ) override;

    /// Tangent operator for a single node of the static beam element
    template <typename VectorView, typename MatrixView>
    void TangentOperator(VectorView psi, MatrixView tangent_operator) {
        static_assert(VectorView::rank == 1);
        static_assert(MatrixView::rank == 2);
        auto populate_matrix = KOKKOS_LAMBDA(size_t) {
            tangent_operator(0, 0) = 1.;
            tangent_operator(1, 1) = 1.;
            tangent_operator(2, 2) = 1.;
            tangent_operator(3, 3) = 1.;
            tangent_operator(4, 4) = 1.;
            tangent_operator(5, 5) = 1.;
        };
        Kokkos::parallel_for(1, populate_matrix);

        const double phi = KokkosBlas::nrm2(psi);
        if (phi > Tolerance) {
            auto psi_cross_prod_matrix = gen_alpha_solver::create_cross_product_matrix(psi);
            auto psi_times_psi = View2D_3x3("psi_times_psi");
            KokkosBlas::gemm(
                "N", "N", 1.0, psi_cross_prod_matrix, psi_cross_prod_matrix, 0.0, psi_times_psi
            );

            auto quadrant4 =
                Kokkos::subview(tangent_operator, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
            auto factor_1 = (std::cos(phi) - 1.0) / (phi * phi);
            auto factor_2 = (1.0 - std::sin(phi) / phi) / (phi * phi);
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
                    {0, 0}, {3, 3}
                ),
                KOKKOS_LAMBDA(const size_t i, const size_t j) {
                    quadrant4(i, j) += factor_1 * psi_cross_prod_matrix(i, j);
                    quadrant4(i, j) += factor_2 * psi_times_psi(i, j);
                }
            );
        }
    }

private:
    View1D position_vectors_;
    StiffnessMatrix stiffness_matrix_;
    UserDefinedQuadrature quadrature_;
};

}  // namespace openturbine::gebt_poc
