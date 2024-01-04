#include "src/gebt_poc/linearization_parameters.h"

namespace openturbine::gebt_poc {

void UnityLinearizationParameters::ResidualVector(
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    [[maybe_unused]] Kokkos::View<double*> lagrange_mults, Kokkos::View<double*> residual_vector,
    [[maybe_unused]] const gen_alpha_solver::TimeStepper& time_stepper
) {
    Kokkos::deep_copy(residual_vector, 0.);
    auto size = residual_vector.extent(0);
    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(size_t i) { residual_vector(i) = 1.; }
    );
}

void UnityLinearizationParameters::IterationMatrix(
    [[maybe_unused]] const double& h, [[maybe_unused]] const double& BETA_PRIME,
    [[maybe_unused]] const double& GAMMA_PRIME,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    [[maybe_unused]] Kokkos::View<double*> lagrange_mults, Kokkos::View<double**> iteration_matrix
) {
    Kokkos::deep_copy(iteration_matrix, 0.);
    auto size = iteration_matrix.extent(0);
    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(size_t i) { iteration_matrix(i, i) = 1.; }
    );
}

}  // namespace openturbine::gebt_poc
