#pragma once

#include "KokkosBlas.hpp"

#include "src/gebt_poc/force.h"
#include "src/gebt_poc/linearization_parameters.h"
#include "src/gebt_poc/solver.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/state.h"
#include "src/gen_alpha_poc/time_stepper.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

/*!
 * Calculates the residual vector and iteration matrix for a dynamic beam element
 */
class DynamicBeamLinearizationParameters : public LinearizationParameters {
public:
    /// Define a dynamic beam element with the given position vector for the nodes, 6 x 6
    /// stiffness matrix, and a quadrature rule
    DynamicBeamLinearizationParameters(
        View1D position_vectors, StiffnessMatrix stiffness_matrix, MassMatrix mass_matrix,
        UserDefinedQuadrature quadrature, std::vector<Forces*> external_forces = {}
    );

    virtual void ResidualVector(
        LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type velocity,
        LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_multipliers,
        const gen_alpha_solver::TimeStepper& time_stepper, View1D residual_vector
    ) override;

    void ApplyExternalForces(double time, const Kokkos::View<double*>& external_forces);

    virtual void IterationMatrix(
        double h, double beta_prime, double gamma_prime, LieGroupFieldView::const_type gen_coords,
        LieAlgebraFieldView::const_type delta_gen_coords, LieAlgebraFieldView::const_type velocity,
        LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_multipliers,
        View2D iteration_matrix
    ) override;

    /// Tangent operator for a single node of the dynamic beam element
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
    View1D position_vectors_;
    StiffnessMatrix stiffness_matrix_;
    MassMatrix mass_matrix_;
    UserDefinedQuadrature quadrature_;
    std::vector<Forces*> external_forces_;
};

}  // namespace openturbine::gebt_poc
