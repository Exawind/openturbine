#include "src/gebt_poc/static_beam_element.h"

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

StaticBeamLinearizationParameters::StaticBeamLinearizationParameters()
    : position_vectors_(Kokkos::View<double*>("position_vectors", 35)),
      stiffness_matrix_(StiffnessMatrix(gen_alpha_solver::create_matrix({
          {1., 2., 3., 4., 5., 6.},       // row 1
          {2., 4., 6., 8., 10., 12.},     // row 2
          {3., 6., 9., 12., 15., 18.},    // row 3
          {4., 8., 12., 16., 20., 24.},   // row 4
          {5., 10., 15., 20., 25., 30.},  // row 5
          {6., 12., 18., 24., 30., 36.}   // row 6
      }))),
      quadrature_(UserDefinedQuadrature(
          std::vector<double>{
              -0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
              0.7415311855993945, 0.9491079123427585},
          std::vector<double>{
              0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
              0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
      )) {
    auto position_vectors = Kokkos::View<double*>("position_vectors", 35);
    auto populate_position_vector = KOKKOS_LAMBDA(size_t) {
        // node 1
        position_vectors(0) = 0.;
        position_vectors(1) = 0.;
        position_vectors(2) = 0.;
        position_vectors(3) = 0.9778215200524469;
        position_vectors(4) = -0.01733607539094763;
        position_vectors(5) = -0.09001900002195001;
        position_vectors(6) = -0.18831121859148398;
        // node 2
        position_vectors(7) = 0.8633658232300573;
        position_vectors(8) = -0.25589826392541715;
        position_vectors(9) = 0.1130411210682743;
        position_vectors(10) = 0.9950113028068008;
        position_vectors(11) = -0.002883848832932071;
        position_vectors(12) = -0.030192109815745303;
        position_vectors(13) = -0.09504013471947484;
        // node 3
        position_vectors(14) = 2.5;
        position_vectors(15) = -0.25;
        position_vectors(16) = 0.;
        position_vectors(17) = 0.9904718430204884;
        position_vectors(18) = -0.009526411091536478;
        position_vectors(19) = 0.09620741150793366;
        position_vectors(20) = 0.09807604012323785;
        // node 4
        position_vectors(21) = 4.136634176769943;
        position_vectors(22) = 0.39875540678255983;
        position_vectors(23) = -0.5416125496397027;
        position_vectors(24) = 0.9472312341234699;
        position_vectors(25) = -0.049692141629315074;
        position_vectors(26) = 0.18127630174800594;
        position_vectors(27) = 0.25965858850765167;
        // node 5
        position_vectors(28) = 5.;
        position_vectors(29) = 1.;
        position_vectors(30) = -1.;
        position_vectors(31) = 0.9210746582719719;
        position_vectors(32) = -0.07193653093139739;
        position_vectors(33) = 0.20507529985516368;
        position_vectors(34) = 0.32309554437664584;
    };
    Kokkos::parallel_for(1, populate_position_vector);
    Kokkos::deep_copy(position_vectors_, position_vectors);
}

StaticBeamLinearizationParameters::StaticBeamLinearizationParameters(
    Kokkos::View<double*> position_vectors, StiffnessMatrix stiffness_matrix,
    UserDefinedQuadrature quadrature
)
    : position_vectors_(position_vectors),
      stiffness_matrix_(stiffness_matrix),
      quadrature_(quadrature) {
}

// TECHDEBT Following is a hack to make things work temporarily - we should move over to
// using 2D views for the solver functions
void Convert2DViewTo1DView(Kokkos::View<double**> view, Kokkos::View<double*> result) {
    auto populate_result = KOKKOS_LAMBDA(size_t i) {
        result(i) = view(i / view.extent(1), i % view.extent(1));
    };
    Kokkos::parallel_for(result.extent(0), populate_result);
}

Kokkos::View<double*> StaticBeamLinearizationParameters::ResidualVector(
    const Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    [[maybe_unused]] const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    const Kokkos::View<double*> lagrange_multipliers
) {
    // The residual vector for the generalized coordinates is given by
    // {residual} = {
    //     {residual_gen_coords},
    //     {residual_constraints}
    // }
    const size_t zero{0};
    const auto size_dofs = velocity.extent(0) * velocity.extent(1);
    const auto size_constraints = lagrange_multipliers.extent(0);
    const auto size_residual = size_dofs + size_constraints;

    auto gen_coords_1D =
        Kokkos::View<double*>("gen_coords_1D", gen_coords.extent(0) * gen_coords.extent(1));
    Convert2DViewTo1DView(gen_coords, gen_coords_1D);

    auto residual = Kokkos::View<double*>("residual", size_residual);
    auto residual_gen_coords = Kokkos::subview(residual, Kokkos::make_pair(zero, size_dofs));
    CalculateStaticResidual(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, residual_gen_coords
    );
    auto residual_constraints =
        Kokkos::subview(residual, Kokkos::make_pair(size_dofs, size_residual));
    ConstraintsResidualVector(gen_coords_1D, position_vectors_, residual_constraints);
    return residual;
}

Kokkos::View<double**> StaticBeamLinearizationParameters::IterationMatrix(
    const double& h, const double& beta_prime, const double& gamma_prime,
    const Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
    const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
    const Kokkos::View<double*> lagrange_mults
) {
}

}  // namespace openturbine::gebt_poc
