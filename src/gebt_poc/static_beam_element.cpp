#include "src/gebt_poc/static_beam_element.h"

#include <KokkosBlas.hpp>

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

// TECHDEBT Following is a hack to make things work temporarily - we should move over to
// using 2D views for the solver functions
void Convert2DViewTo1DView(View2D::const_type view, View1D result) {
    auto populate_result = KOKKOS_LAMBDA(size_t i) {
        result(i) = view(i / view.extent(1), i % view.extent(1));
    };
    Kokkos::parallel_for(result.extent(0), populate_result);
}

void BMatrix(View2D constraints_gradient_matrix) {
    // Assemble the constraint gradient matrix i.e. B matrix
    // [B]_6x(n+1) = [
    //     [I]_3x3        [0]       [0]   ....  [0]
    //        [0]       [I]_3x3     [0]   ....  [0]
    // ]
    // where
    // [I]_3x3 = [1]_3x3
    // [0] = [0]_3x3

    Kokkos::deep_copy(constraints_gradient_matrix, 0.);
    auto B11 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3)
    );
    auto B22 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6)
    );

    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            B11(0, 0) = 1.;
            B11(1, 1) = 1.;
            B11(2, 2) = 1.;
            B22(0, 0) = 1.;
            B22(1, 1) = 1.;
            B22(2, 2) = 1.;
        }
    );
}

struct DefinePositionVector_5NodeBeamElement {
    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
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
    }
    Kokkos::View<double[35]> position_vectors_;
};

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
    Kokkos::parallel_for(1, DefinePositionVector_5NodeBeamElement{position_vectors_});
}

StaticBeamLinearizationParameters::StaticBeamLinearizationParameters(
    View1D position_vectors, StiffnessMatrix stiffness_matrix, UserDefinedQuadrature quadrature
)
    : position_vectors_(position_vectors),
      stiffness_matrix_(stiffness_matrix),
      quadrature_(quadrature) {
}

void StaticBeamLinearizationParameters::ResidualVector(
    LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type velocity,
    [[maybe_unused]] LieAlgebraFieldView::const_type acceleration,
    View1D::const_type lagrange_multipliers,
    [[maybe_unused]] const gen_alpha_solver::TimeStepper& time_stepper, View1D residual
) {
    // The residual vector for the generalized coordinates is given by
    // {residual} = {
    //     {residual_gen_coords} + {constraints_part2},
    //     {residual_constraints}
    // }
    const size_t zero{0};
    const auto size_dofs = velocity.extent(0) * velocity.extent(1);
    const auto size_constraints = lagrange_multipliers.extent(0);
    const auto size_residual = size_dofs + size_constraints;

    // Part 1: Calculate the residual vector for the generalized coordinates
    auto gen_coords_1D = View1D("gen_coords_1D", gen_coords.extent(0) * gen_coords.extent(1));
    Convert2DViewTo1DView(gen_coords, gen_coords_1D);

    Kokkos::deep_copy(residual, 0.0);
    auto residual_gen_coords = Kokkos::subview(residual, Kokkos::make_pair(zero, size_dofs));
    ElementalStaticForcesResidual(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, residual_gen_coords
    );

    // Part 2: Calculate the residual vector for the constraints
    // {R_c} = {B(q)}^T * {lambda}
    auto constraints_gradient_matrix =
        View2D("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    auto constraints_part2 = View1D("constraints_part2", size_dofs);
    KokkosBlas::gemv(
        "T", 1., constraints_gradient_matrix, lagrange_multipliers, 0., constraints_part2
    );

    KokkosBlas::axpy(1., constraints_part2, residual_gen_coords);

    auto residual_constraints =
        Kokkos::subview(residual, Kokkos::make_pair(size_dofs, size_residual));
    ElementalConstraintForcesResidual(gen_coords_1D, residual_constraints);
}

void StaticBeamLinearizationParameters::IterationMatrix(
    double h, [[maybe_unused]] double beta_prime, [[maybe_unused]] double gamma_prime,
    LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type delta_gen_coords,
    LieAlgebraFieldView::const_type velocity,
    [[maybe_unused]] LieAlgebraFieldView::const_type acceleration,
    View1D::const_type lagrange_multipliers, View2D iteration_matrix
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
    const auto size_iteration = size_dofs + size_constraints;
    const auto n_nodes = velocity.extent(0);

    auto gen_coords_1D = View1D("gen_coords_1D", gen_coords.extent(0) * gen_coords.extent(1));
    Convert2DViewTo1DView(gen_coords, gen_coords_1D);

    // Assemble the tangent operator (same size as the stiffness matrix)
    auto delta_gen_coords_node = View1D("delta_gen_coords_node", VectorComponents);
    auto tangent_operator = View2D("tangent_operator", size_dofs, size_dofs);
    Kokkos::deep_copy(tangent_operator, 0.0);
    for (size_t i = 0; i < n_nodes; ++i) {
        Kokkos::deep_copy(
            delta_gen_coords_node, Kokkos::subview(delta_gen_coords, i, Kokkos::make_pair(3, 6))
        );
        KokkosBlas::scal(delta_gen_coords_node, h, delta_gen_coords_node);
        auto tangent_operator_node = Kokkos::subview(
            tangent_operator,
            Kokkos::make_pair(i * LieAlgebraComponents, (i + 1) * LieAlgebraComponents),
            Kokkos::make_pair(i * LieAlgebraComponents, (i + 1) * LieAlgebraComponents)
        );
        TangentOperator(delta_gen_coords_node, tangent_operator_node);
    }

    Kokkos::deep_copy(iteration_matrix, 0.0);

    // Calculate the beam element static iteration matrix
    auto iteration_matrix_local = View2D("iteration_matrix_local", size_dofs, size_dofs);
    ElementalStaticStiffnessMatrix(
        position_vectors_, gen_coords_1D, stiffness_matrix_, quadrature_, iteration_matrix_local
    );

    // Combine beam element static iteration matrix with constraints into quadrant 1
    // quadrant_1 = K_t + K_t_part2
    auto quadrant_1 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs), Kokkos::make_pair(zero, size_dofs)
    );
    KokkosBlas::gemm("N", "N", 1.0, iteration_matrix_local, tangent_operator, 0.0, quadrant_1);

    // quadrant_2 = Transpose([B(q)])
    auto quadrant_2 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(zero, size_dofs),
        Kokkos::make_pair(size_dofs, size_iteration)
    );
    auto constraints_gradient_matrix =
        View2D("constraints_gradient_matrix", size_constraints, size_dofs);
    BMatrix(constraints_gradient_matrix);
    auto temp = gen_alpha_solver::transpose_matrix(constraints_gradient_matrix);
    Kokkos::deep_copy(quadrant_2, temp);

    // quadrant_3 = B(q) * T(h dq)
    auto quadrant_3 = Kokkos::subview(
        iteration_matrix, Kokkos::make_pair(size_dofs, size_iteration),
        Kokkos::make_pair(zero, size_dofs)
    );
    KokkosBlas::gemm("N", "N", 1.0, constraints_gradient_matrix, tangent_operator, 0.0, quadrant_3);
}

}  // namespace openturbine::gebt_poc
