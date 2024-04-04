#include "beams_methods.hpp"

#include <array>

#include "beams_functors.hpp"

namespace openturbine {

struct UpdateNodeState {
    Kokkos::View<size_t*> node_state_indices;
    View_Nx7 node_u;
    View_Nx6 node_u_dot;
    View_Nx6 node_u_ddot;

    View_Nx7 Q;
    View_Nx6 V;
    View_Nx6 A;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        auto j = node_state_indices(i);
        for (size_t k = 0; k < kLieGroupComponents; k++) {
            node_u(i, k) = Q(j, k);
        }
        for (size_t k = 0; k < kLieAlgebraComponents; k++) {
            node_u_dot(i, k) = V(j, k);
        }
        for (size_t k = 0; k < kLieAlgebraComponents; k++) {
            node_u_ddot(i, k) = A(j, k);
        }
    }
};

// Update node states (displacement, velocity, acceleration) and interpolate to quadrature points
void UpdateState(Beams& beams, View_Nx7 Q, View_Nx6 V, View_Nx6 A) {
    // Copy displacement, velocity, and acceleration to nodes
    Kokkos::parallel_for(
        "UpdateNodeState", beams.num_nodes,
        UpdateNodeState{
            beams.node_state_indices, beams.node_u, beams.node_u_dot, beams.node_u_ddot, Q, V, A}
    );

    // Interpolate node state to quadrature points
    Kokkos::parallel_for(
        "InterpolateQPState", beams.num_elems,
        InterpolateQPState{
            beams.elem_indices, beams.shape_interp, beams.shape_deriv, beams.qp_jacobian,
            beams.node_u, beams.qp_u, beams.qp_u_prime, beams.qp_r, beams.qp_r_prime}
    );
    Kokkos::parallel_for(
        "InterpolateQPVelocity", beams.num_elems,
        InterpolateQPVelocity{
            beams.elem_indices, beams.shape_interp, beams.shape_deriv, beams.qp_jacobian,
            beams.node_u_dot, beams.qp_u_dot, beams.qp_omega}
    );
    Kokkos::parallel_for(
        "InterpolateQPAcceleration", beams.num_elems,
        InterpolateQPAcceleration{
            beams.elem_indices, beams.shape_interp, beams.shape_deriv, beams.qp_jacobian,
            beams.node_u_ddot, beams.qp_u_ddot, beams.qp_omega_dot}
    );

    // Calculate RR0 matrix
    Kokkos::parallel_for(
        "CalculateRR0", beams.num_qps,
        CalculateRR0{
            beams.qp_r0,
            beams.qp_r,
            beams.qp_quat,
            beams.qp_RR0,
        }
    );

    // Calculate Muu matrix
    Kokkos::parallel_for(
        "CalculateMuu", beams.num_qps,
        CalculateMuu{beams.qp_RR0, beams.qp_Mstar, beams.qp_Muu, beams.M_6x6}
    );

    // Calculate Cuu matrix
    Kokkos::parallel_for(
        "CalculateCuu", beams.num_qps,
        CalculateCuu{beams.qp_RR0, beams.qp_Cstar, beams.qp_Cuu, beams.M_6x6}
    );

    // Calculate strain
    Kokkos::parallel_for(
        "CalculateStrain", beams.num_qps,
        CalculateStrain{
            beams.qp_x0_prime,
            beams.qp_u_prime,
            beams.qp_r,
            beams.qp_r_prime,
            beams.M_3x4,
            beams.V1_3,
            beams.qp_strain,
        }
    );

    // Calculate Forces
    Kokkos::parallel_for(
        "CalculateMassMatrixComponents", beams.num_qps,
        CalculateMassMatrixComponents{beams.qp_Muu, beams.V1_3, beams.M7_3x3, beams.M1_3x3}
    );
    Kokkos::parallel_for(
        "CalculateTemporaryVariables", beams.num_qps,
        CalculateTemporaryVariables{beams.qp_x0_prime, beams.qp_u_prime, beams.V2_3, beams.M4_3x3}
    );
    Kokkos::parallel_for(
        "CalculateForceFC", beams.num_qps,
        CalculateForceFC{beams.qp_Cuu, beams.qp_strain, beams.qp_Fc, beams.M5_3x3, beams.M6_3x3}
    );
    Kokkos::parallel_for(
        "CalculateForceFD", beams.num_qps, CalculateForceFD{beams.M4_3x3, beams.qp_Fc, beams.qp_Fd}
    );
    Kokkos::parallel_for(
        "CalculateInertialForces", beams.num_qps,
        CalculateInertialForces{
            beams.qp_Muu, beams.qp_u_ddot, beams.qp_omega, beams.qp_omega_dot, beams.M1_3x3,
            beams.M2_3x3, beams.M3_3x3, beams.M7_3x3, beams.V1_3, beams.V2_3, beams.M8_3x3,
            beams.qp_Fi}
    );
    Kokkos::parallel_for(
        "CalculateGravityForce", beams.num_qps,
        CalculateGravityForce{beams.gravity, beams.qp_Muu, beams.M1_3x3, beams.V2_3, beams.qp_Fg}
    );
    Kokkos::parallel_for(
        "CalculateOuu", beams.num_qps,
        CalculateOuu{beams.qp_Cuu, beams.M4_3x3, beams.M5_3x3, beams.M6_3x3, beams.qp_Ouu}
    );
    Kokkos::parallel_for(
        "CalculatePuu", beams.num_qps,
        CalculatePuu{beams.qp_Cuu, beams.M4_3x3, beams.M6_3x3, beams.qp_Puu}
    );
    Kokkos::parallel_for(
        "CalculateQuu", beams.num_qps,
        CalculateQuu{beams.qp_Cuu, beams.M4_3x3, beams.M6_3x3, beams.M8_3x3, beams.qp_Quu}
    );
    Kokkos::parallel_for(
        "CalculateGyroscopicMatrix", beams.num_qps,
        CalculateGyroscopicMatrix{
            beams.qp_Muu, beams.qp_omega, beams.M2_3x3, beams.M7_3x3, beams.V1_3, beams.V2_3,
            beams.V3_3, beams.M8_3x3, beams.qp_Guu}
    );
    Kokkos::parallel_for(
        "CalculateInertiaStiffnessMatrix", beams.num_qps,
        CalculateInertiaStiffnessMatrix{
            beams.qp_Muu, beams.qp_u_ddot, beams.qp_omega, beams.qp_omega_dot, beams.M2_3x3,
            beams.M3_3x3, beams.M7_3x3, beams.V1_3, beams.V2_3, beams.M8_3x3, beams.M9_3x3,
            beams.qp_Kuu}
    );

    // Calculate nodal force vectors
    Kokkos::parallel_for(
        "CalculateNodeForces", beams.num_elems,
        CalculateNodeForces{
            beams.elem_indices, beams.qp_weight, beams.qp_jacobian, beams.shape_interp,
            beams.shape_deriv, beams.qp_Fc, beams.qp_Fd, beams.qp_Fi, beams.qp_Fg, beams.node_FE,
            beams.node_FI, beams.node_FG}
    );
}

void AssembleResidualVector(Beams& beams, View_N residual_vector) {
    Kokkos::parallel_for(
        beams.num_nodes, IntegrateResidualVector(
                             beams.node_state_indices, beams.node_FE, beams.node_FI, beams.node_FG,
                             beams.node_FX, residual_vector
                         )
    );
}

void AssembleMassMatrix(Beams& beams, View_NxN M) {
    Kokkos::parallel_for(
        "IntegrateMatrix", beams.num_elems,
        IntegrateMatrix{
            beams.elem_indices,
            beams.node_state_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.qp_Muu,
            M,
        }
    );
}

void AssembleGyroscopicInertiaMatrix(Beams& beams, View_NxN G) {
    Kokkos::parallel_for(
        "IntegrateMatrix", beams.num_elems,
        IntegrateMatrix{
            beams.elem_indices,
            beams.node_state_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.qp_Guu,
            G,
        }
    );
}

void AssembleInertialStiffnessMatrix(Beams& beams, View_NxN K) {
    Kokkos::parallel_for(
        "IntegrateMatrix", beams.num_elems,
        IntegrateMatrix{
            beams.elem_indices,
            beams.node_state_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.qp_Kuu,
            K,
        }
    );
}

void AssembleElasticStiffnessMatrix(Beams& beams, View_NxN K) {
    Kokkos::parallel_for(
        "IntegrateElasticStiffnessMatrix", beams.num_elems,
        IntegrateElasticStiffnessMatrix{
            beams.elem_indices,
            beams.node_state_indices,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.shape_deriv,
            beams.qp_Puu,
            beams.qp_Cuu,
            beams.qp_Ouu,
            beams.qp_Quu,
            K,
        }
    );
}

}  // namespace openturbine