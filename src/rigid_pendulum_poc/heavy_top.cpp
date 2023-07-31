#include "src/rigid_pendulum_poc/heavy_top.h"

#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

HostView1D heavy_top_residual_vector(
    HostView2D mass_matrix, HostView2D rotation_matrix, HostView1D acceleration_vector,
    HostView1D gen_forces_vector, HostView1D position_vector, HostView1D lagrange_multipliers,
    HostView1D reference_position_vector
) {
    // The residual vector for the generalized coordinates is given by
    // {residual} = {
    //     {residual_gen_coords},
    //     {residual_constraints}
    // }

    auto residual_gen_coords = heavy_top_gen_coords_residual_vector(
        mass_matrix, rotation_matrix, acceleration_vector, gen_forces_vector,
        reference_position_vector, lagrange_multipliers
    );

    auto residual_constraints =
        heavy_top_constraints_residual_vector(rotation_matrix, position_vector);

    auto size_res_gen_coords = residual_gen_coords.extent(0);
    auto size_res_constraints = residual_constraints.extent(0);

    auto size = size_res_gen_coords + size_res_constraints;
    auto residual_vector = HostView1D("residual_vector", size);
    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const size_t i) {
            if (i < size_res_gen_coords) {
                residual_vector(i) = residual_gen_coords(i);
            }
            residual_vector(i) = residual_constraints(i - 6);
        }
    );

    return residual_vector;
}

HostView1D heavy_top_gen_coords_residual_vector(
    HostView2D mass_matrix, HostView2D rotation_matrix, HostView1D acceleration_vector,
    HostView1D gen_forces_vector, HostView1D lagrange_multipliers,
    HostView1D reference_position_vector
) {
    // The residual vector for the generalized coordinates is given by
    // {residual_gen_coords} = [M(q)] {v'} + {g(q,v,t)} + [B(q)]T {Lambda}
    // where,
    // [M(q)] = mass matrix
    // {v'} = acceleration vector
    // {g(q,v,t)} = generalized forces vector
    // [B(q)] = constraint gradient matrix
    // {Lambda} = Lagrange multipliers vector

    // Calculate residual vector for the generalized coordinates
    auto constraint_gradient_matrix =
        heavy_top_constraint_gradient_matrix(rotation_matrix, reference_position_vector);

    auto first_term = multiply_matrix_with_vector(mass_matrix, acceleration_vector);
    auto second_term = gen_forces_vector;
    auto third_term = multiply_matrix_with_vector(
        transpose_matrix(constraint_gradient_matrix), lagrange_multipliers
    );

    auto residual_gen_coords = HostView1D("residual_gen_coords", 6);
    Kokkos::parallel_for(
        6, KOKKOS_LAMBDA(const size_t i
           ) { residual_gen_coords(i) = first_term(i) + second_term(i) + third_term(i); }
    );

    auto log = util::Log::Get();
    log->Debug("Residual vector is " + std::to_string(6) + " x 1 with elements\n");
    for (size_t i = 0; i < 6; i++) {
        log->Debug(std::to_string(residual_gen_coords(i)) + "\n");
    }

    return residual_gen_coords;
}

HostView1D heavy_top_constraints_residual_vector(
    HostView2D rotation_matrix, HostView1D position_vector, HostView1D reference_position_vector
) {
    // The residual vector for the constraints is given by
    // {residual_constraints} = -{x} + [R] {X}
    // where,
    // {x} = position vector
    // [R] = rotation matrix
    // {X} = reference position vector

    auto RX = multiply_matrix_with_vector(rotation_matrix, reference_position_vector);

    auto residual_constraints = HostView1D("constraint_residual_vector", 3);
    Kokkos::parallel_for(
        3, KOKKOS_LAMBDA(const size_t i) { residual_constraints(i) = -position_vector(i) + RX(i); }
    );

    auto log = util::Log::Get();
    log->Debug("Residual vector is " + std::to_string(3) + " x 1 with elements\n");
    for (size_t i = 0; i < 3; i++) {
        log->Debug(std::to_string(residual_constraints(i)) + "\n");
    }

    return residual_constraints;
}

HostView2D heavy_top_constraint_gradient_matrix(
    HostView2D rotation_matrix, HostView1D reference_position_vector
) {
    // Constraint gradient matrix for the heavy top problem is given by
    // [B] = [ -I_3x3    -[R ~{X}] ]
    auto I_3x3 = create_identity_matrix(3);

    auto X = create_cross_product_matrix(reference_position_vector);
    auto RX = multiply_matrix_with_matrix(rotation_matrix, X);

    auto constraint_gradient_matrix = HostView2D("constraint_gradient_matrix", 3, 6);
    Kokkos::parallel_for(
        3,
        KOKKOS_LAMBDA(const size_t i) {
            for (size_t j = 0; j < 6; j++) {
                if (j < 3) {
                    constraint_gradient_matrix(i, j) = -I_3x3(i, j);
                } else {
                    constraint_gradient_matrix(i, j) = -RX(i, j - 3);
                }
            }
        }
    );

    auto log = util::Log::Get();
    log->Debug(
        "Constraint gradient matrix is " + std::to_string(3) + " x " + std::to_string(6) +
        " with elements" + "\n"
    );
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 6; j++) {
            log->Debug(
                "(" + std::to_string(i) + ", " + std::to_string(j) +
                ") : " + std::to_string(constraint_gradient_matrix(i, j)) + "\n"
            );
        }
    }

    return constraint_gradient_matrix;
}

HostView2D heavy_top_iteration_matrix(
    const double& BETA_PRIME, const double& GAMMA_PRIME, HostView2D mass_matrix,
    HostView2D inertia_matrix, HostView2D rotation_matrix, HostView1D angular_velocity_vector,
    HostView1D lagrange_multipliers, HostView1D reference_position_vector, double h,
    HostView1D delta_gen_coords
) {
    // Iteration matrix for the heavy top problem is given by
    // [iteration matrix] = [
    //     [M(q)] * beta' + [C_t(q,v,t)] * gamma' + [K_t(q,v,v',Lambda,t)] * [T(h dq)]    [B(q)^T]]
    //                         [ B(q) ] * [T(h dq)]                                         [0]
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
    auto tangent_stiffness_matrix = heavy_top_tangent_stiffness_matrix(
        rotation_matrix, lagrange_multipliers, reference_position_vector
    );
    auto constraint_gradient_matrix =
        heavy_top_constraint_gradient_matrix(rotation_matrix, reference_position_vector);

    auto h_delta_gen_coords = HostView1D("h_delta_gen_coords", 3);
    Kokkos::parallel_for(
        3, KOKKOS_LAMBDA(const size_t i) { h_delta_gen_coords(i) = h * delta_gen_coords(i + 3); }
    );
    auto tangent_operator = heavy_top_tangent_operator(h_delta_gen_coords);

    auto size_dofs = mass_matrix.extent(0);
    auto size_constraints = constraint_gradient_matrix.extent(0);
    auto size = size_dofs + size_constraints;

    auto element1 = HostView2D("element1", size_dofs, size_dofs);
    auto K_T_hdq = multiply_matrix_with_matrix(tangent_stiffness_matrix, tangent_operator);
    Kokkos::parallel_for(
        size_dofs,
        KOKKOS_LAMBDA(const size_t i) {
            for (size_t j = 0; j < size_dofs; j++) {
                element1(i, j) = mass_matrix(i, j) * BETA_PRIME +
                                 tangent_damping_matrix(i, j) * GAMMA_PRIME + K_T_hdq(i, j);
            }
        }
    );
    auto element2 = transpose_matrix(constraint_gradient_matrix);

    auto B_T_hdq = multiply_matrix_with_matrix(constraint_gradient_matrix, tangent_operator);
    auto element3 = B_T_hdq;

    auto element4 = HostView2D("element4", 3, 3);

    auto iteration_matrix = HostView2D("iteration_matrix", size, size);
    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const size_t i) {
            for (size_t j = 0; j < size; j++) {
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

    auto log = util::Log::Get();
    log->Debug(
        "Iteration matrix is " + std::to_string(size) + " x " + std::to_string(size) +
        " with elements" + "\n"
    );
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            log->Debug(
                "(" + std::to_string(i) + ", " + std::to_string(j) +
                ") : " + std::to_string(iteration_matrix(i, j)) + "\n"
            );
        }
    }

    return iteration_matrix;
}

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
        KOKKOS_LAMBDA(const size_t i) {
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
        KOKKOS_LAMBDA(const size_t i) {
            for (size_t j = 0; j < 6; j++) {
                if (i >= 3 && j >= 3) {
                    tangent_damping_matrix(i, j) = nonzero_block(i - 3, j - 3);
                }
            }
        }
    );

    auto log = util::Log::Get();
    log->Debug(
        "Tangent damping matrix is " + std::to_string(6) + " x " + std::to_string(6) +
        " with elements" + "\n"
    );
    for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 6; j++) {
            log->Debug(
                "(" + std::to_string(i) + ", " + std::to_string(j) +
                ") : " + std::to_string(tangent_damping_matrix(i, j)) + "\n"
            );
        }
    }

    return tangent_damping_matrix;
}

HostView2D heavy_top_tangent_stiffness_matrix(
    HostView2D rotation_matrix, HostView1D lagrange_multipliers, HostView1D reference_position_vector
) {
    // Tangent stiffness matrix for the heavy top problem is given by
    // [K_t] = [ [0]_3x3              [0]_3x3
    //           [0]_3x3    [ ~{X} * ~([R^T] * {Lambda}) ] ]
    auto X = create_cross_product_matrix(reference_position_vector);

    auto RT_Lambda =
        multiply_matrix_with_vector(transpose_matrix(rotation_matrix), lagrange_multipliers);
    auto RT_Lambda_matrix = create_cross_product_matrix(RT_Lambda);

    auto non_zero_block = multiply_matrix_with_matrix(X, RT_Lambda_matrix);

    // Only the 3 x 3 lower right block of the tangent stiffness matrix is non-zero
    auto tangent_stiffness_matrix = HostView2D("tangent_stiffness_matrix", 6, 6);
    Kokkos::parallel_for(
        6,
        KOKKOS_LAMBDA(const size_t i) {
            for (size_t j = 0; j < 6; j++) {
                if (i >= 3 && j >= 3) {
                    tangent_stiffness_matrix(i, j) = non_zero_block(i - 3, j - 3);
                }
            }
        }
    );

    auto log = util::Log::Get();
    log->Debug(
        "Tangent stiffness matrix is " + std::to_string(6) + " x " + std::to_string(6) +
        " with elements" + "\n"
    );
    for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 6; j++) {
            log->Debug(
                "(" + std::to_string(i) + ", " + std::to_string(j) +
                ") : " + std::to_string(tangent_stiffness_matrix(i, j)) + "\n"
            );
        }
    }

    return tangent_stiffness_matrix;
}

HostView2D heavy_top_tangent_operator(HostView1D psi) {
    const double tol = 1e-16;
    const double phi = std::sqrt(psi(0) * psi(0) + psi(1) * psi(1) + psi(2) * psi(2));

    auto tangent_operator = HostView2D("tangent_operator", 6, 6);
    tangent_operator(0, 0) = 1.0;
    tangent_operator(1, 1) = 1.0;
    tangent_operator(2, 2) = 1.0;
    tangent_operator(3, 3) = 1.0;
    tangent_operator(4, 4) = 1.0;
    tangent_operator(5, 5) = 1.0;

    if (std::abs(phi) > tol) {
        auto psi_matrix = create_cross_product_matrix(psi);
        auto psi_psi_matrix = multiply_matrix_with_matrix(psi_matrix, psi_matrix);

        auto tangent_operator_1 =
            multiply_matrix_with_scalar(psi_matrix, (std::cos(phi) - 1.0) / (phi * phi));
        auto tangent_operator_2 =
            multiply_matrix_with_scalar(psi_psi_matrix, (1.0 - std::sin(phi) / phi) / (phi * phi));

        Kokkos::parallel_for(
            6,
            KOKKOS_LAMBDA(const size_t i) {
                for (size_t j = 0; j < 6; j++) {
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

HostView2D rigid_pendulum_iteration_matrix(size_t size) {
    // TODO: Implement this
    return create_identity_matrix(size);
}

HostView1D rigid_pendulum_residual_vector(size_t size) {
    // TODO: Implement this
    return create_identity_vector(size);
}

}  // namespace openturbine::rigid_pendulum
