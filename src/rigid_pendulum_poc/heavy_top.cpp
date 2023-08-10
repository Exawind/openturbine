#include "src/rigid_pendulum_poc/heavy_top.h"

#include "src/rigid_pendulum_poc/quaternion.h"
#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

HeavyTopLinearizationParameters::HeavyTopLinearizationParameters() {
    this->mass_matrix_ = MassMatrix(15., Vector(0.234375, 0.46875, 0.234375));
}

HostView2D HeavyTopLinearizationParameters::CalculateRotationMatrix(const HostView1D gen_coords) {
    // Convert the quaternion representing orientation -> rotation matrix
    auto RM = quaternion_to_rotation_matrix(
        // Create quaternion from appropriate components of generalized coordinates
        Quaternion{gen_coords(3), gen_coords(4), gen_coords(5), gen_coords(6)}
    );
    auto [m00, m01, m02] = std::get<0>(RM).GetComponents();
    auto [m10, m11, m12] = std::get<1>(RM).GetComponents();
    auto [m20, m21, m22] = std::get<2>(RM).GetComponents();
    auto rotation_matrix = create_matrix({{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}});

    return rotation_matrix;
}

HostView1D HeavyTopLinearizationParameters::CalculateForces(
    MassMatrix mass_matrix, const HostView1D velocity
) {
    // Generalized forces as defined in Brüls and Cardona 2010
    auto mass = 15.;
    auto gravity = Vector(0., 0., 9.81);
    auto forces = gravity * mass;

    auto angular_velocity = create_vector({
        velocity(3),  // velocity component 4 -> component 1
        velocity(4),  // velocity component 5 -> component 2
        velocity(5)   // velocity component 6 -> component 3
    });
    auto J = mass_matrix.GetMomentOfInertiaMatrix();
    auto J_omega = multiply_matrix_with_vector(J, angular_velocity);

    auto angular_velocity_vector =
        Vector(angular_velocity(0), angular_velocity(1), angular_velocity(2));
    auto J_omega_vector = Vector(J_omega(0), J_omega(1), J_omega(2));
    auto moments = angular_velocity_vector.CrossProduct(J_omega_vector);

    auto generalized_forces = GeneralizedForces(forces, moments);

    return generalized_forces.GetGeneralizedForces();
}

HostView1D HeavyTopLinearizationParameters::ResidualVector(
    const HostView1D gen_coords, const HostView1D velocity, const HostView1D acceleration,
    const HostView1D lagrange_multipliers
) {
    // The residual vector for the generalized coordinates is given by
    // {residual} = {
    //     {residual_gen_coords},
    //     {residual_constraints}
    // }

    if (gen_coords.extent(0) != 7) {
        throw std::invalid_argument("gen_coords must be of size 7");
    }

    if (velocity.extent(0) != 6 || acceleration.extent(0) != 6) {
        throw std::invalid_argument("delta_gen_coords, velocity, acceleration must be of size 6");
    }

    auto mass_matrix = this->mass_matrix_.GetMassMatrix();
    auto rotation_matrix = CalculateRotationMatrix(gen_coords);
    auto gen_forces_vector = CalculateForces(this->mass_matrix_, velocity);
    auto position_vector = create_vector(
        // Create vector from appropriate components of generalized coordinates
        {gen_coords(0), gen_coords(1), gen_coords(2)}
    );
    const auto reference_position_vector = create_vector({0., 1., 0});

    auto residual_gen_coords = GeneralizedCoordinatesResidualVector(
        mass_matrix, rotation_matrix, acceleration, gen_forces_vector, lagrange_multipliers,
        reference_position_vector
    );
    auto residual_constraints =
        ConstraintsResidualVector(rotation_matrix, position_vector, reference_position_vector);

    auto size_res_gen_coords = residual_gen_coords.extent(0);
    auto size_res_constraints = residual_constraints.extent(0);
    auto size = size_res_gen_coords + size_res_constraints;
    auto residual_vector = HostView1D("residual_vector", size);
    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const size_t i) {
            if (i >= size_res_gen_coords) {
                residual_vector(i) = residual_constraints(i - size_res_gen_coords);
            } else {
                residual_vector(i) = residual_gen_coords(i);
            }
        }
    );

    return residual_vector;
}

HostView1D HeavyTopLinearizationParameters::GeneralizedCoordinatesResidualVector(
    const HostView2D mass_matrix, const HostView2D rotation_matrix,
    const HostView1D acceleration_vector, const HostView1D gen_forces_vector,
    const HostView1D lagrange_multipliers, const HostView1D reference_position_vector
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
        ConstraintsGradientMatrix(rotation_matrix, reference_position_vector);

    auto first_term = multiply_matrix_with_vector(mass_matrix, acceleration_vector);

    auto second_term = HostView1D("second_term", 6);
    Kokkos::deep_copy(second_term, gen_forces_vector);

    auto third_term = multiply_matrix_with_vector(
        transpose_matrix(constraint_gradient_matrix), lagrange_multipliers
    );

    auto residual_gen_coords = HostView1D("residual_gen_coords", 6);
    // clang-format off
    Kokkos::parallel_for(
        6,
        KOKKOS_LAMBDA(const size_t i) {
           residual_gen_coords(i) = first_term(i) + second_term(i) + third_term(i);
        }
    );
    // clang-format on

    return residual_gen_coords;
}

HostView1D HeavyTopLinearizationParameters::ConstraintsResidualVector(
    const HostView2D rotation_matrix, const HostView1D position_vector,
    const HostView1D reference_position_vector
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

    return residual_constraints;
}

HostView2D HeavyTopLinearizationParameters::ConstraintsGradientMatrix(
    const HostView2D rotation_matrix, const HostView1D reference_position_vector
) {
    // Constraint gradient matrix for the heavy top problem is given by
    // [B] = [ -I_3x3    -[R ~{X}] ]
    auto I_3x3 = create_identity_matrix(3);

    auto X = create_cross_product_matrix(reference_position_vector);
    auto RX = multiply_matrix_with_matrix(rotation_matrix, X);

    auto constraint_gradient_matrix = HostView2D("constraint_gradient_matrix", 3, 6);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>({0, 0}, {3, 6}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (j < 3) {
                constraint_gradient_matrix(i, j) = -I_3x3(i, j);
            } else {
                constraint_gradient_matrix(i, j) = -RX(i, j - 3);
            }
        }
    );

    return constraint_gradient_matrix;
}

HostView2D HeavyTopLinearizationParameters::IterationMatrix(
    const double& h, const double& BETA_PRIME, const double& GAMMA_PRIME,
    const HostView1D gen_coords, const HostView1D delta_gen_coords, const HostView1D velocity,
    [[maybe_unused]] const HostView1D acceleration, const HostView1D lagrange_mults
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

    if (gen_coords.extent(0) != 7) {
        throw std::invalid_argument("gen_coords must be of size 7");
    }

    if (delta_gen_coords.extent(0) != 6 || velocity.extent(0) != 6 || acceleration.extent(0) != 6) {
        throw std::invalid_argument("delta_gen_coords, velocity, acceleration must be of size 6");
    }

    auto mass_matrix = this->mass_matrix_.GetMassMatrix();
    auto moment_of_inertia_matrix = this->mass_matrix_.GetMomentOfInertiaMatrix();
    auto rotation_matrix = CalculateRotationMatrix(gen_coords);
    auto gen_forces_vector = CalculateForces(this->mass_matrix_, velocity);
    auto angular_velocity_vector = create_vector({velocity(3), velocity(4), velocity(5)});
    auto position_vector = create_vector({gen_coords(0), gen_coords(1), gen_coords(2)});
    const HostView1D reference_position_vector = create_vector({0., 1., 0});

    auto tangent_damping_matrix =
        TangentDampingMatrix(angular_velocity_vector, moment_of_inertia_matrix);
    auto tangent_stiffness_matrix =
        TangentStiffnessMatrix(rotation_matrix, lagrange_mults, reference_position_vector);
    auto constraint_gradient_matrix =
        ConstraintsGradientMatrix(rotation_matrix, reference_position_vector);

    auto h_delta_gen_coords = HostView1D("h_delta_gen_coords", 3);
    Kokkos::parallel_for(
        3, KOKKOS_LAMBDA(const size_t i) { h_delta_gen_coords(i) = h * delta_gen_coords(i + 3); }
    );
    auto tangent_operator = TangentOperator(h_delta_gen_coords);

    auto size_dofs = mass_matrix.extent(0);
    auto size_constraints = constraint_gradient_matrix.extent(0);
    auto size = size_dofs + size_constraints;

    auto element1 = HostView2D("element1", size_dofs, size_dofs);
    auto K_T_hdq = multiply_matrix_with_matrix(tangent_stiffness_matrix, tangent_operator);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {size_dofs, size_dofs}
        ),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i < size_dofs && j < size_dofs) {
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
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {size, size}
        ),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
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
    );

    return iteration_matrix;
}

HostView2D HeavyTopLinearizationParameters::TangentDampingMatrix(
    const HostView1D angular_velocity_vector, const HostView2D inertia_matrix
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
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>({0, 0}, {3, 3}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i < 3 && j < 3) {
                nonzero_block(i, j) =
                    nonzero_block_first_part(i, j) - nonzero_block_second_part(i, j);
            }
        }
    );

    // Only the 3 x 3 lower right block of the tangent damping matrix is non-zero
    auto tangent_damping_matrix = HostView2D("tangent_damping_matrix", 6, 6);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>({0, 0}, {6, 6}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i >= 3 && j >= 3) {
                tangent_damping_matrix(i, j) = nonzero_block(i - 3, j - 3);
            }
        }
    );

    return tangent_damping_matrix;
}

HostView2D HeavyTopLinearizationParameters::TangentStiffnessMatrix(
    const HostView2D rotation_matrix, const HostView1D lagrange_multipliers,
    const HostView1D reference_position_vector
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
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>({0, 0}, {6, 6}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i >= 3 && j >= 3) {
                tangent_stiffness_matrix(i, j) = non_zero_block(i - 3, j - 3);
            }
        }
    );

    return tangent_stiffness_matrix;
}

HostView2D HeavyTopLinearizationParameters::TangentOperator(const HostView1D psi) {
    const double tol = 1e-16;
    const double phi = std::sqrt(psi(0) * psi(0) + psi(1) * psi(1) + psi(2) * psi(2));

    auto tangent_operator = HostView2D("tangent_operator", 6, 6);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>({0, 0}, {6, 6}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i == j) {
                tangent_operator(i, j) = 1.0;
            }
        }
    );

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

}  // namespace openturbine::rigid_pendulum
