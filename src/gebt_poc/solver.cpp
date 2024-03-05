#include "src/gebt_poc/solver.h"

#include <KokkosBlas.hpp>

#include "src/gebt_poc/element.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/NodalCurvature.hpp"
#include "src/gebt_poc/CalculateSectionalStrain.hpp"
#include "src/gebt_poc/SectionalStiffness.hpp"
#include "src/gebt_poc/NodalElasticForces.hpp"
#include "src/gebt_poc/SectionalMassMatrix.hpp"
#include "src/gebt_poc/NodalInertialForces.hpp"
#include "src/gebt_poc/NodalStaticStiffnessMatrixComponents.hpp"
#include "src/gebt_poc/NodalGyroscopicMatrix.hpp"
#include "src/gebt_poc/NodalDynamicStiffnessMatrix.hpp"

namespace openturbine::gebt_poc {

void ElementalConstraintForcesResidual(View1D::const_type gen_coords, View1D constraints_residual) {
    Kokkos::deep_copy(constraints_residual, 0.);
    // For the GEBT proof of concept problem (i.e. the clamped beam), the dofs are enforced to be
    // zero at the left end of the beam, so the constraint residual is simply based on the
    // generalized coordinates at the first node
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            // Construct rotation vector from root node rotation quaternion
            auto rotation_vector = openturbine::gen_alpha_solver::rotation_vector_from_quaternion(
                openturbine::gen_alpha_solver::Quaternion(
                    gen_coords(3), gen_coords(4), gen_coords(5), gen_coords(6)
                )
            );
            // Set residual as translation and rotation of root node
            // TODO: update when position & rotations are prescribed
            constraints_residual(0) = gen_coords(0);
            constraints_residual(1) = gen_coords(1);
            constraints_residual(2) = gen_coords(2);
            constraints_residual(3) = rotation_vector.GetXComponent();
            constraints_residual(4) = rotation_vector.GetYComponent();
            constraints_residual(5) = rotation_vector.GetZComponent();
        }
    );
}

void ElementalConstraintForcesResidual(
    LieGroupFieldView::const_type gen_coords, View1D constraints_residual
) {
    Kokkos::deep_copy(constraints_residual, 0.);
    // For the GEBT proof of concept problem (i.e. the clamped beam), the dofs are enforced to be
    // zero at the left end of the beam, so the constraint residual is simply based on the
    // generalized coordinates at the first node
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            // Construct rotation vector from root node rotation quaternion
            auto rotation_vector = openturbine::gen_alpha_solver::rotation_vector_from_quaternion(
                openturbine::gen_alpha_solver::Quaternion(
                    gen_coords(0, 3), gen_coords(0, 4), gen_coords(0, 5), gen_coords(0, 6)
                )
            );
            // Set residual as translation and rotation of root node
            // TODO: update when position & rotations are prescribed
            constraints_residual(0) = gen_coords(0, 0);
            constraints_residual(1) = gen_coords(0, 1);
            constraints_residual(2) = gen_coords(0, 2);
            constraints_residual(3) = rotation_vector.GetXComponent();
            constraints_residual(4) = rotation_vector.GetYComponent();
            constraints_residual(5) = rotation_vector.GetZComponent();
        }
    );
}

void ElementalConstraintForcesGradientMatrix(
    View1D::const_type gen_coords, View1D::const_type position_vector,
    View2D constraints_gradient_matrix
) {
    auto translation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(0, 3));
    auto rotation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(3, 7));
    auto rotation_matrix_0 = gen_alpha_solver::EulerParameterToRotationMatrix(rotation_0);
    auto position_0 = Kokkos::subview(position_vector, Kokkos::make_pair(0, 3));

    // position_cross_prod_matrix = ~{position_0} + ~{translation_0}
    auto position_0_cross_prod_matrix = gen_alpha_solver::create_cross_product_matrix(position_0);
    auto translation_0_cross_prod_matrix =
        gen_alpha_solver::create_cross_product_matrix(translation_0);
    auto position_cross_prod_matrix = View2D_3x3("position_cross_prod_matrix");
    Kokkos::deep_copy(position_cross_prod_matrix, position_0_cross_prod_matrix);
    KokkosBlas::axpy(1., translation_0_cross_prod_matrix, position_cross_prod_matrix);

    // Assemble the constraint gradient matrix i.e. B matrix for the beam element
    // [B]_6x(n+1) = [
    //     [B11]_3x3              0            0   ....  0
    //     [B21]_3x3          [B22]_3x3        0   ....  0
    // ]
    // where
    // [B11]_3x3 = [1]_3x3
    // [B21]_3x3 = -[rotation_matrix_0]_3x3
    // [B22]_3x3 = -[rotation_matrix_0]_3x3 * [position_cross_prod_matrix]_3x3
    // n = order of the element
    Kokkos::deep_copy(constraints_gradient_matrix, 0.);
    auto B11 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3)
    );
    auto B21 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3)
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
        }
    );

    KokkosBlas::scal(B21, -1., rotation_matrix_0);
    KokkosBlas::gemm("N", "N", -1., rotation_matrix_0, position_cross_prod_matrix, 0., B22);
}

}  // namespace openturbine::gebt_poc
