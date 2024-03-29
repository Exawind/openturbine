#include "src/gebt_poc/linearization_parameters.h"

namespace openturbine::gebt_poc {

void UnityLinearizationParameters::ResidualVector(
    [[maybe_unused]] LieGroupFieldView::const_type gen_coords,
    [[maybe_unused]] LieAlgebraFieldView::const_type velocity,
    [[maybe_unused]] LieAlgebraFieldView::const_type acceleration,
    [[maybe_unused]] View1D::const_type lagrange_mults,
    [[maybe_unused]] const gen_alpha_solver::TimeStepper& time_stepper, View1D residual_vector
) {
    Kokkos::deep_copy(residual_vector, 0.);
    auto size = residual_vector.extent(0);
    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(size_t i) { residual_vector(i) = 1.; }
    );
}

void UnityLinearizationParameters::IterationMatrix(
    [[maybe_unused]] double h, [[maybe_unused]] double BETA_PRIME,
    [[maybe_unused]] double GAMMA_PRIME, [[maybe_unused]] LieGroupFieldView::const_type gen_coords,
    [[maybe_unused]] LieAlgebraFieldView::const_type delta_gen_coords,
    [[maybe_unused]] LieAlgebraFieldView::const_type velocity,
    [[maybe_unused]] LieAlgebraFieldView::const_type acceleration,
    [[maybe_unused]] View1D::const_type lagrange_mults, View2D iteration_matrix
) {
    Kokkos::deep_copy(iteration_matrix, 0.);
    auto size = iteration_matrix.extent(0);
    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(size_t i) { iteration_matrix(i, i) = 1.; }
    );
}

}  // namespace openturbine::gebt_poc
