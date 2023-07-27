#include "src/rigid_pendulum_poc/heavy_top.h"

namespace openturbine::rigid_pendulum {

HostView2D heavy_top_tangent_damping_matrix(
    HostView1D angular_velocity_vector, HostView2D inertia_matrix
) {
    // Tangent damping matrix for the heavy top problem is given by
    // [C_t] = [ [0]_3x3                     [0]_3x3
    //           [0]_3x3    [ ~{OMEGA}] * [J] - ~([J] * {OMEGA}) ]
    auto angular_velocity_matrix = create_cross_product_matrix(angular_velocity_vector);

    auto nonzero_block_first_part =
        multiply_matrix_with_matrix(angular_velocity_matrix, inertia_matrix);

    auto J_Omega = multiply_matrix_with_vector(inertia_matrix, angular_velocity_vector);
    auto nonzero_block_second_part = create_cross_product_matrix(J_Omega);

    auto nonzero_block = HostView2D("nonzero_block", 3, 3);
    Kokkos::parallel_for(
        3,
        KOKKOS_LAMBDA(const int i) {
            for (size_t j = 0; j < 3; j++) {
                nonzero_block(i, j) =
                    nonzero_block_first_part(i, j) - nonzero_block_second_part(i, j);
            }
        }
    );

    // Only the 3 x 3 lower right block of the tangent damping matrix is non-zero
    auto tangent_damping_matrix = HostView2D("tangent_damping_matrix", 6, 6);
    Kokkos::parallel_for(
        6,
        KOKKOS_LAMBDA(const int i) {
            for (size_t j = 0; j < 6; j++) {
                if (i < 3 && j < 3) {
                    tangent_damping_matrix(i, j) = 0.;
                } else if (i < 3 && j >= 3) {
                    tangent_damping_matrix(i, j) = 0.;
                } else if (i >= 3 && j < 3) {
                    tangent_damping_matrix(i, j) = 0.;
                } else {
                    tangent_damping_matrix(i, j) = nonzero_block(i - 3, j - 3);
                }
            }
        }
    );

    return tangent_damping_matrix;
}

HostView2D heavy_top_tangent_stiffness_matrix(
    HostView1D position_vector, HostView2D rotation_matrix, HostView1D lagrange_multipliers
) {
    // Tangent stiffness matrix for the heavy top problem is given by
    // [K_t] = [ [0]_3x3              [0]_3x3
    //           [0]_3x3    [ ~{X} * ~([R^T] * {Lambda}) ] ]
    auto X = create_cross_product_matrix(position_vector);

    auto RT_Lambda =
        multiply_matrix_with_vector(transpose_matrix(rotation_matrix), lagrange_multipliers);
    auto RT_Lambda_matrix = create_cross_product_matrix(RT_Lambda);

    auto non_zero_block = multiply_matrix_with_matrix(X, RT_Lambda_matrix);

    // Only the 3 x 3 lower right block of the tangent stiffness matrix is non-zero
    auto tangent_stiffness_matrix = HostView2D("tangent_stiffness_matrix", 6, 6);
    Kokkos::parallel_for(
        6,
        KOKKOS_LAMBDA(const int i) {
            for (size_t j = 0; j < 6; j++) {
                if (i < 3 && j < 3) {
                    tangent_stiffness_matrix(i, j) = 0.;
                } else if (i < 3 && j >= 3) {
                    tangent_stiffness_matrix(i, j) = 0.;
                } else if (i >= 3 && j < 3) {
                    tangent_stiffness_matrix(i, j) = 0.;
                } else {
                    tangent_stiffness_matrix(i, j) = non_zero_block(i - 3, j - 3);
                }
            }
        }
    );

    return tangent_stiffness_matrix;
}

HostView2D heavy_top_constraint_gradient_matrix(
    HostView1D position_vector, HostView2D rotation_matrix
) {
    // Constraint gradient matrix for the heavy top problem is given by
    // [B] = [ -I_3x3    -[R ~{X}] ]
    auto I_3x3 = create_identity_matrix(3);

    auto X = create_cross_product_matrix(position_vector);
    auto RX = multiply_matrix_with_matrix(rotation_matrix, X);

    auto constraint_gradient_matrix = HostView2D("constraint_gradient_matrix", 3, 6);
    Kokkos::parallel_for(
        3,
        KOKKOS_LAMBDA(const int i) {
            for (size_t j = 0; j < 6; j++) {
                if (j < 3) {
                    constraint_gradient_matrix(i, j) = -I_3x3(i, j);
                } else {
                    constraint_gradient_matrix(i, j) = -RX(i, j - 3);
                }
            }
        }
    );

    return constraint_gradient_matrix;
}

HostView2D heavy_top_iteration_matrix(
    HostView2D mass_matrix, HostView2D inertia_matrix, HostView2D rotation_matrix,
    HostView1D angular_velocity_vector, HostView1D position_vector, HostView1D lagrange_multipliers,
    const double BETA_PRIME, const double GAMMA_PRIME
) {
    // Iteration matrix for the heavy top problem is given by
    // [iteration matrix] = [
    //     [M(q)] * beta' + [C_t(q,v,t)] * gamma' + [K_t(q,v,v',Lambda,t)]    [B(q)^T]]
    //                            [ B(q) ]                                       [0]
    // ]
    // where,
    // [M(q)] = mass matrix
    // [C_t(q,v,t)] = Tangent damping matrix = [ 0            0
    //                                           0    OMEGA * J - J * OMEGA ]
    // [K_t(q,v,v',Lambda,t)] = Tangent stiffness matrix = [ 0          0
    //                                                       0  X * R^T * Lambda ]
    // [B(q)] = Constraint gradeint matrix = [ -I_3    -R * X ]

    auto tangent_damping_matrix =
        heavy_top_tangent_damping_matrix(angular_velocity_vector, inertia_matrix);
    auto tangent_stiffness_matrix =
        heavy_top_tangent_stiffness_matrix(position_vector, rotation_matrix, lagrange_multipliers);
    auto constraint_matrix = heavy_top_constraint_gradient_matrix(position_vector, rotation_matrix);

    auto size_dofs = mass_matrix.extent(0);
    auto size_constraints = constraint_matrix.extent(0);
    auto size_it_matrix = size_dofs + size_constraints;

    auto element1 = HostView2D("element1", size_dofs, size_dofs);
    Kokkos::parallel_for(
        size_dofs,
        KOKKOS_LAMBDA(const int i) {
            for (size_t j = 0; j < size_dofs; j++) {
                element1(i, j) = mass_matrix(i, j) * BETA_PRIME +
                                 tangent_damping_matrix(i, j) * GAMMA_PRIME +
                                 tangent_stiffness_matrix(i, j);
            }
        }
    );
    auto element2 = transpose_matrix(constraint_matrix);
    auto element3 = constraint_matrix;
    auto element4 = HostView2D("element4", 3, 3);

    auto iteration_matrix = HostView2D("iteration_matrix", size_it_matrix, size_it_matrix);
    Kokkos::parallel_for(
        size_it_matrix,
        KOKKOS_LAMBDA(const size_t i) {
            for (size_t j = 0; j < size_it_matrix; j++) {
                if (i < size_dofs && j < size_dofs) {
                    iteration_matrix(i, j) = element1(i, j);
                } else if (i < size_dofs && j >= size_dofs) {
                    iteration_matrix(i, j) = element2(i, j - size_dofs);
                } else if (i >= size_dofs && j < size_dofs) {
                    iteration_matrix(i, j) = element3(i - size_dofs, j);
                } else {
                    iteration_matrix(i, j) = element4(i - size_dofs, j - size_dofs);
                }
            }
        }
    );

    return iteration_matrix;
}

HostView2D rigid_pendulum_iteration_matrix(size_t size) {
    // TODO: Implement this
    return create_identity_matrix(size);
}

}  // namespace openturbine::rigid_pendulum
