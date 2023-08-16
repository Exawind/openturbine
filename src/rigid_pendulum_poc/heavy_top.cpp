#include "src/rigid_pendulum_poc/heavy_top.h"

#include "src/rigid_pendulum_poc/quaternion.h"
#include "src/rigid_pendulum_poc/rotation_matrix.h"
#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

HeavyTopLinearizationParameters::HeavyTopLinearizationParameters() {
    this->mass_matrix_ = MassMatrix(15., Vector(0.234375, 0.46875, 0.234375));
}

Kokkos::View<double**> HeavyTopLinearizationParameters::CalculateRotationMatrix(const Kokkos::View<double*> gen_coords) {
    auto gen_coords_host = Kokkos::create_mirror(gen_coords);
    Kokkos::deep_copy(gen_coords_host, gen_coords);
    // Convert the quaternion representing orientation -> rotation matrix
    auto RM = quaternion_to_rotation_matrix(
        // Create quaternion from appropriate components of generalized coordinates
        Quaternion{gen_coords_host(3), gen_coords_host(4), gen_coords_host(5), gen_coords_host(6)}
    );
    auto rotation_matrix = create_matrix({{RM(0,0), RM(0,1), RM(0,2)}, 
                                          {RM(1,0), RM(1,1), RM(1,2)},
                                          {RM(2,0), RM(2,1), RM(2,2)}});

    return rotation_matrix;
}

Kokkos::View<double*> HeavyTopLinearizationParameters::CalculateForces(
    MassMatrix mass_matrix, const Kokkos::View<double*> velocity
) {
    // Generalized forces as defined in Brüls and Cardona 2010
    auto mass = 15.;
    auto gravity = Vector(0., 0., 9.81);
    auto forces = gravity * mass;

    auto velocity_host = Kokkos::create_mirror(velocity);
    Kokkos::deep_copy(velocity_host, velocity);

    auto angular_velocity = create_vector({
        velocity_host(3),  // velocity component 4 -> component 1
        velocity_host(4),  // velocity component 5 -> component 2
        velocity_host(5)   // velocity component 6 -> component 3
    });
    auto J = mass_matrix.GetMomentOfInertiaMatrix();
    auto J_omega = multiply_matrix_with_vector(J, angular_velocity);

    auto angular_velocity_host = Kokkos::create_mirror(angular_velocity);
    Kokkos::deep_copy(angular_velocity_host, angular_velocity);

    auto J_omega_host = Kokkos::create_mirror(J_omega);
    Kokkos::deep_copy(J_omega_host, J_omega);

    auto angular_velocity_vector =
        Vector(angular_velocity_host(0), angular_velocity_host(1), angular_velocity_host(2));
    auto J_omega_vector = Vector(J_omega_host(0), J_omega_host(1), J_omega_host(2));
    auto moments = angular_velocity_vector.CrossProduct(J_omega_vector);

    auto generalized_forces = GeneralizedForces(forces, moments);

    return generalized_forces.GetGeneralizedForces();
}

Kokkos::View<double*> HeavyTopLinearizationParameters::ResidualVector(
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> velocity, const Kokkos::View<double*> acceleration,
    const Kokkos::View<double*> lagrange_multipliers
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

    auto gen_coords_host = Kokkos::create_mirror(gen_coords);
    Kokkos::deep_copy(gen_coords_host, gen_coords);

    auto position_vector = create_vector(
        // Create vector from appropriate components of generalized coordinates
        {gen_coords_host(0), gen_coords_host(1), gen_coords_host(2)}
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
    auto residual_vector = Kokkos::View<double*>("residual_vector", size);
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

Kokkos::View<double*> HeavyTopLinearizationParameters::GeneralizedCoordinatesResidualVector(
    const Kokkos::View<double**> mass_matrix, const Kokkos::View<double**> rotation_matrix,
    const Kokkos::View<double*> acceleration_vector, const Kokkos::View<double*> gen_forces_vector,
    const Kokkos::View<double*> lagrange_multipliers, const Kokkos::View<double*> reference_position_vector
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

    auto second_term = Kokkos::View<double*>("second_term", 6);
    Kokkos::deep_copy(second_term, gen_forces_vector);

    auto third_term = multiply_matrix_with_vector(
        transpose_matrix(constraint_gradient_matrix), lagrange_multipliers
    );

    auto residual_gen_coords = Kokkos::View<double*>("residual_gen_coords", 6);
    Kokkos::parallel_for(
        6, KOKKOS_LAMBDA(const size_t i
           ) { residual_gen_coords(i) = first_term(i) + second_term(i) + third_term(i); }
    );

    return residual_gen_coords;
}

Kokkos::View<double*> HeavyTopLinearizationParameters::ConstraintsResidualVector(
    const Kokkos::View<double**> rotation_matrix, const Kokkos::View<double*> position_vector,
    const Kokkos::View<double*> reference_position_vector
) {
    // The residual vector for the constraints is given by
    // {residual_constraints} = -{x} + [R] {X}
    // where,
    // {x} = position vector
    // [R] = rotation matrix
    // {X} = reference position vector

    auto RX = multiply_matrix_with_vector(rotation_matrix, reference_position_vector);

    auto residual_constraints = Kokkos::View<double*>("constraint_residual_vector", 3);
    Kokkos::parallel_for(
        3, KOKKOS_LAMBDA(const size_t i) { residual_constraints(i) = -position_vector(i) + RX(i); }
    );

    return residual_constraints;
}

Kokkos::View<double**> HeavyTopLinearizationParameters::ConstraintsGradientMatrix(
    const Kokkos::View<double**> rotation_matrix, const Kokkos::View<double*> reference_position_vector
) {
    // Constraint gradient matrix for the heavy top problem is given by
    // [B] = [ -I_3x3    -[R ~{X}] ]
    auto I_3x3 = create_identity_matrix(3);

    auto X = create_cross_product_matrix(reference_position_vector);
    auto RX = multiply_matrix_with_matrix(rotation_matrix, X);

    auto constraint_gradient_matrix = Kokkos::View<double**>("constraint_gradient_matrix", 3, 6);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {3, 6}),
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

Kokkos::View<double**> HeavyTopLinearizationParameters::IterationMatrix(
    const double& h, const double& BETA_PRIME, const double& GAMMA_PRIME,
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> delta_gen_coords, const Kokkos::View<double*> velocity,
    [[maybe_unused]] const Kokkos::View<double*> acceleration, const Kokkos::View<double*> lagrange_mults
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
    auto velocity_host = Kokkos::create_mirror(velocity);
    Kokkos::deep_copy(velocity_host, velocity);

    auto gen_coords_host = Kokkos::create_mirror(gen_coords);
    Kokkos::deep_copy(gen_coords_host, gen_coords);

    auto angular_velocity_vector = create_vector({velocity_host(3), velocity_host(4), velocity_host(5)});
    auto position_vector = create_vector({gen_coords_host(0), gen_coords_host(1), gen_coords_host(2)});
    const Kokkos::View<double*> reference_position_vector = create_vector({0., 1., 0});

    auto tangent_damping_matrix =
        TangentDampingMatrix(angular_velocity_vector, moment_of_inertia_matrix);
    auto tangent_stiffness_matrix =
        TangentStiffnessMatrix(rotation_matrix, lagrange_mults, reference_position_vector);
    auto constraint_gradient_matrix =
        ConstraintsGradientMatrix(rotation_matrix, reference_position_vector);

    auto h_delta_gen_coords = Kokkos::View<double*>("h_delta_gen_coords", 3);
    Kokkos::parallel_for(
        3, KOKKOS_LAMBDA(const size_t i) { h_delta_gen_coords(i) = h * delta_gen_coords(i + 3); }
    );
    auto tangent_operator = TangentOperator(h_delta_gen_coords);

    auto size_dofs = mass_matrix.extent(0);
    auto size_constraints = constraint_gradient_matrix.extent(0);
    auto size = size_dofs + size_constraints;

    auto element1 = Kokkos::View<double**>("element1", size_dofs, size_dofs);
    auto K_T_hdq = multiply_matrix_with_matrix(tangent_stiffness_matrix, tangent_operator);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
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

    auto element4 = Kokkos::View<double**>("element4", 3, 3);

    auto iteration_matrix = Kokkos::View<double**>("iteration_matrix", size, size);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
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

Kokkos::View<double**> HeavyTopLinearizationParameters::TangentDampingMatrix(
    const Kokkos::View<double*> angular_velocity_vector, const Kokkos::View<double**> inertia_matrix
) {
    // Tangent damping matrix for the heavy top problem is given by
    // [C_t] = [ [0]_3x3                     [0]_3x3
    //           [0]_3x3    [ ~{OMEGA}] * [J] - ~([J] * {OMEGA}) ]
    auto angular_velocity_matrix = create_cross_product_matrix(angular_velocity_vector);

    auto nonzero_block_first_part =
        multiply_matrix_with_matrix(angular_velocity_matrix, inertia_matrix);

    auto J_Omega = multiply_matrix_with_vector(inertia_matrix, angular_velocity_vector);
    auto nonzero_block_second_part = create_cross_product_matrix(J_Omega);

    auto nonzero_block = Kokkos::View<double**>("nonzero_block", 3, 3);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {3, 3}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i < 3 && j < 3) {
                nonzero_block(i, j) =
                    nonzero_block_first_part(i, j) - nonzero_block_second_part(i, j);
            }
        }
    );

    // Only the 3 x 3 lower right block of the tangent damping matrix is non-zero
    auto tangent_damping_matrix = Kokkos::View<double**>("tangent_damping_matrix", 6, 6);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {6, 6}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i >= 3 && j >= 3) {
                tangent_damping_matrix(i, j) = nonzero_block(i - 3, j - 3);
            }
        }
    );

    return tangent_damping_matrix;
}

Kokkos::View<double**> HeavyTopLinearizationParameters::TangentStiffnessMatrix(
    const Kokkos::View<double**> rotation_matrix, const Kokkos::View<double*> lagrange_multipliers,
    const Kokkos::View<double*> reference_position_vector
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
    auto tangent_stiffness_matrix = Kokkos::View<double**>("tangent_stiffness_matrix", 6, 6);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {6, 6}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            if (i >= 3 && j >= 3) {
                tangent_stiffness_matrix(i, j) = non_zero_block(i - 3, j - 3);
            }
        }
    );

    return tangent_stiffness_matrix;
}

Kokkos::View<double**> HeavyTopLinearizationParameters::TangentOperator(const Kokkos::View<double*> psi) {
    const double tol = 1e-16;
    
    auto psi_host = Kokkos::create_mirror(psi);
    Kokkos::deep_copy(psi_host, psi);
    const double phi = std::sqrt(psi_host(0) * psi_host(0) + psi_host(1) * psi_host(1) + psi_host(2) * psi_host(2));

    auto tangent_operator = Kokkos::View<double**>("tangent_operator", 6, 6);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {6, 6}),
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
