#include "src/gebt_poc/static_beam_element.h"

#include <KokkosBlas.hpp>

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

StaticBeamLinearizationParameters::StaticBeamLinearizationParameters()
    : position_vectors_(Kokkos::View<double[35]>("position_vectors")),
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
              -0.9491079123427585,  // point 1
              -0.7415311855993945,  // point 2
              -0.4058451513773972,  // point 3
              0.,                   // point 4
              0.4058451513773972,   // point 5
              0.7415311855993945,   // point 6
              0.9491079123427585    // point 7
          },
          std::vector<double>{
              0.1294849661688697,  // weight 1
              0.2797053914892766,  // weight 2
              0.3818300505051189,  // weight 3
              0.4179591836734694,  // weight 4
              0.3818300505051189,  // weight 5
              0.2797053914892766,  // weight 6
              0.1294849661688697   // weight 7
          }
      )) {
    // Define the position vectors for the 5 node beam element
    auto populate_position_vector = KOKKOS_LAMBDA(size_t) {
        // node 1
        position_vectors_(0) = 0.;
        position_vectors_(1) = 0.;
        position_vectors_(2) = 0.;
        position_vectors_(3) = 0.9778215200524469;
        position_vectors_(4) = -0.01733607539094763;
        position_vectors_(5) = -0.09001900002195001;
        position_vectors_(6) = -0.18831121859148398;
        // node 2
        position_vectors_(7) = 0.8633658232300573;
        position_vectors_(8) = -0.25589826392541715;
        position_vectors_(9) = 0.1130411210682743;
        position_vectors_(10) = 0.9950113028068008;
        position_vectors_(11) = -0.002883848832932071;
        position_vectors_(12) = -0.030192109815745303;
        position_vectors_(13) = -0.09504013471947484;
        // node 3
        position_vectors_(14) = 2.5;
        position_vectors_(15) = -0.25;
        position_vectors_(16) = 0.;
        position_vectors_(17) = 0.9904718430204884;
        position_vectors_(18) = -0.009526411091536478;
        position_vectors_(19) = 0.09620741150793366;
        position_vectors_(20) = 0.09807604012323785;
        // node 4
        position_vectors_(21) = 4.136634176769943;
        position_vectors_(22) = 0.39875540678255983;
        position_vectors_(23) = -0.5416125496397027;
        position_vectors_(24) = 0.9472312341234699;
        position_vectors_(25) = -0.049692141629315074;
        position_vectors_(26) = 0.18127630174800594;
        position_vectors_(27) = 0.25965858850765167;
        // node 5
        position_vectors_(28) = 5.;
        position_vectors_(29) = 1.;
        position_vectors_(30) = -1.;
        position_vectors_(31) = 0.9210746582719719;
        position_vectors_(32) = -0.07193653093139739;
        position_vectors_(33) = 0.20507529985516368;
        position_vectors_(34) = 0.32309554437664584;
    };
    Kokkos::parallel_for(1, populate_position_vector);
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
    const Kokkos::View<double*> lagrange_multipliers
) {
    // Iteration matrix for the static beam element is given by
    // [iteration matrix] = [
    //     [K_t(q,v,v',Lambda,t)] * [T(h dq)]                   [B(q)^T]]
    //          [ B(q) ] * [T(h dq)]                               [0]
    // ]
    // where,
    // [K_t(q,v,v',Lambda,t)] = Tangent stiffness matrix
    // [T(h dq)] = Tangent operator
    // [B(q)] = Constraint gradient matrix
    const size_t zero{0};
    const auto size_dofs = velocity.extent(0) * velocity.extent(1);
    const auto size_constraints = lagrange_multipliers.extent(0);
    const auto size = size_dofs + size_constraints;

    auto gen_coords_1D =
        Kokkos::View<double*>("gen_coords_1D", gen_coords.extent(0) * gen_coords.extent(1));
    Convert2DViewTo1DView(gen_coords, gen_coords_1D);

    auto iteration_matrix = Kokkos::View<double**>("iteration_matrix", size, size);
    Kokkos::deep_copy(iteration_matrix, 0.0);

    auto quadrant_1 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs), Kokkos::make_pair(zero, size_dofs)
    );
    CalculateStaticIterationMatrix(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, quadrant_1
    );
    // TODO: Add tangent operator

    auto quadrant_2 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(size_dofs, size), Kokkos::make_pair(zero, size_dofs)
    );
    ConstraintsGradientMatrix(gen_coords_1D, position_vectors_, quadrant_2);

    auto quadrant_3 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs), Kokkos::make_pair(size_dofs, size)
    );
    Kokkos::deep_copy(quadrant_3, 0.0);
    // TODO: Add tangent operator

    auto quadrant_4 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(size_dofs, size), Kokkos::make_pair(size_dofs, size)
    );
}

// auto log = util::Log::Get();
// log->Debug("tanget_operator:\n");
// for (size_t i = 0; i < kNumberOfLieAlgebraComponents; ++i) {
//     for (size_t j = 0; j < kNumberOfLieAlgebraComponents; ++j) {
//         log->Debug(std::to_string(tangent_operator(i, j)) + "\n");
//     }
//     log->Debug("\n");
// }

Kokkos::View<double**> StaticBeamLinearizationParameters::TangentOperator(
    const Kokkos::View<double*> psi
) {
    auto tangent_operator = Kokkos::View<double**>(
        "tangent_operator", kNumberOfLieAlgebraComponents, kNumberOfLieAlgebraComponents
    );
    Kokkos::deep_copy(tangent_operator, 0.);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {kNumberOfLieAlgebraComponents, kNumberOfLieAlgebraComponents}
        ),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i == j) {
                tangent_operator(i, j) = 1.0;
            }
        }
    );

    auto psi_host = Kokkos::create_mirror(psi);
    Kokkos::deep_copy(psi_host, psi);
    const double phi =
        std::sqrt(psi_host(0) * psi_host(0) + psi_host(1) * psi_host(1) + psi_host(2) * psi_host(2));

    // TODO clean up the following to use subviews and Blas calls
    if (std::abs(phi) > kTolerance) {
        auto psi_cross_prod_matrix = gen_alpha_solver::create_cross_product_matrix(psi);
        auto psi_times_psi = gen_alpha_solver::multiply_matrix_with_matrix(
            psi_cross_prod_matrix, psi_cross_prod_matrix
        );

        auto tangent_operator_1 = gen_alpha_solver::multiply_matrix_with_scalar(
            psi_times_psi, (std::cos(phi) - 1.0) / (phi * phi)
        );
        auto tangent_operator_2 = gen_alpha_solver::multiply_matrix_with_scalar(
            psi_times_psi, (1.0 - std::sin(phi) / phi) / (phi * phi)
        );

        Kokkos::parallel_for(
            kNumberOfLieAlgebraComponents,
            KOKKOS_LAMBDA(const size_t i) {
                for (size_t j = 0; j < kNumberOfLieAlgebraComponents; j++) {
                    if (i >= 3 && j >= 3) {
                        tangent_operator(i, j) += tangent_operator_1(i - 3, j - 3);
                        tangent_operator(i, j) += tangent_operator_2(i - 3, j - 3);
                    }
                }
            }
        );
    }

    return tangent_operator;
}

}  // namespace openturbine::gebt_poc
