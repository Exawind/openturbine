#pragma once

#include <array>
#include <numeric>

#include <KokkosBlas.hpp>

#include "types.hpp"

#include "src/gebt_poc/quadrature.h"

namespace oturb {

struct BeamElemIndices {
    int num_nodes;
    int num_qps;
    Kokkos::pair<int, int> node_range;
    Kokkos::pair<int, int> qp_range;
};

//------------------------------------------------------------------------------
// Beam input data structures
//------------------------------------------------------------------------------

struct BeamNode {
    double s;
    std::array<double, 3> x;
    std::array<double, 4> r;
    BeamNode(std::array<double, 3> x_, std::array<double, 4> r_)
        : s(0.), x(std::move(x_)), r(std::move(r_)) {}
};

struct BeamSection {
    double s;
    std::array<std::array<double, 6>, 6> M_star;
    std::array<std::array<double, 6>, 6> C_star;
    BeamSection(
        double s_, std::array<std::array<double, 6>, 6> M_star_,
        std::array<std::array<double, 6>, 6> C_star_
    )
        : s(s_), M_star(std::move(M_star_)), C_star(std::move(C_star_)) {}
};

struct BeamInput {
    openturbine::gebt_poc::UserDefinedQuadrature quadrature;
    std::vector<BeamNode> nodes;
    std::vector<BeamSection> sections;

    BeamInput(
        openturbine::gebt_poc::UserDefinedQuadrature quadrature_, std::vector<BeamNode> nodes_,
        std::vector<BeamSection> sections_
    )
        : quadrature(std::move(quadrature_)),
          nodes(std::move(nodes_)),
          sections(std::move(sections_)) {
        //
        // Calculate distances between nodes in element
        std::vector<double> node_distances({0.});
        for (size_t i = 1; i < this->nodes.size(); i++) {
            node_distances.push_back(sqrt(
                pow(this->nodes[i].x[0] - this->nodes[i - 1].x[0], 2) +
                pow(this->nodes[i].x[1] - this->nodes[i - 1].x[1], 2) +
                pow(this->nodes[i].x[2] - this->nodes[i - 1].x[2], 2)
            ));
        }

        // Calculate total element length
        double length = std::reduce(node_distances.begin(), node_distances.end());

        // Calculate cumulate distance of nodes
        std::vector<double> node_cumulative_distances(node_distances.size());
        std::partial_sum(
            node_distances.begin(), node_distances.end(), node_cumulative_distances.begin()
        );

        for (size_t i = 0; i < this->nodes.size(); i++) {
            this->nodes[i].s = node_cumulative_distances[i] / length;
        }
    }
};

//------------------------------------------------------------------------------
// Functors to perform calculations on Beams structure
//------------------------------------------------------------------------------

struct InterpolateQPPosition {
    Kokkos::View<BeamElemIndices*> elem_indices;  // Element indices
    oturb::View_NxN shape_interp_;                // Num Nodes x Num Quadrature points
    oturb::View_Nx7 node_pos_rot_;                // Node global position vector
    oturb::View_Nx3 qp_pos_;                      // quadrature point position

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        // Element specific views
        auto shape_interp = Kokkos::subview(
            shape_interp_, elem_indices[i_elem].node_range,
            Kokkos::make_pair(0, elem_indices[i_elem].num_qps)
        );
        auto node_pos =
            Kokkos::subview(node_pos_rot_, elem_indices[i_elem].node_range, Kokkos::make_pair(0, 3));
        auto qp_pos = Kokkos::subview(qp_pos_, elem_indices[i_elem].qp_range, Kokkos::ALL);

        // Initialize qp_pos
        Kokkos::deep_copy(qp_pos, 0.);

        // Perform matrix-matrix multiplication
        for (int i = 0; i < elem_indices[i_elem].num_nodes; ++i) {
            for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
                for (int k = 0; k < 3; ++k) {
                    qp_pos(j, k) += node_pos(i, k) * shape_interp(i, j);
                }
            }
        }
    }
};

KOKKOS_FUNCTION
void InterpMatMul3(View_NxN shape_matrix, View_Nx3 node_v, View_Nx3 qp_v);

KOKKOS_FUNCTION
void InterpMatMul4(View_NxN shape_matrix, View_Nx4 node_v, View_Nx4 qp_v);

KOKKOS_FUNCTION
void InterpQuaternion(View_NxN shape_matrix, View_Nx4 node_v, View_Nx4 qp_v);

KOKKOS_FUNCTION
void InterpDeriv3(View_NxN shape_matrix, View_N jacobian, View_Nx3 node_v, View_Nx3 qp_v);

KOKKOS_FUNCTION
void InterpDeriv4(View_NxN shape_matrix, View_N jacobian, View_Nx4 node_v, View_Nx4 qp_v);

struct InterpolateQPState {
    Kokkos::View<BeamElemIndices*> elem_indices;  // Element indices

    oturb::View_NxN shape_interp_;  // Num Nodes x Num Quadrature points
    oturb::View_NxN shape_deriv_;   // Num Nodes x Num Quadrature points
    oturb::View_N qp_jacobian_;     // Num Nodes x Num Quadrature points
    oturb::View_Nx7 node_u_;        // Node translation & rotation displacement
    oturb::View_Nx6 node_u_dot_;    // Node translation & angular velocity
    oturb::View_Nx6 node_u_ddot_;   // Node translation & angular acceleration

    oturb::View_Nx3 qp_u_;          // qp translation displacement
    oturb::View_Nx4 qp_r_;          // qp rotation displacement
    oturb::View_Nx3 qp_u_dot_;      // qp translation velocity
    oturb::View_Nx3 qp_omega_;      // qp angular velocity
    oturb::View_Nx3 qp_u_ddot_;     // qp translation acceleration
    oturb::View_Nx3 qp_omega_dot_;  // qp angular acceleration

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        // Element specific views
        auto& idx = elem_indices[i_elem];
        auto shape_interp =
            Kokkos::subview(shape_interp_, idx.node_range, Kokkos::make_pair(0, idx.num_qps));
        auto shape_deriv =
            Kokkos::subview(shape_deriv_, idx.node_range, Kokkos::make_pair(0, idx.num_qps));

        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);

        // Interpolate translation displacement
        auto node_u = Kokkos::subview(node_u_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_u = Kokkos::subview(qp_u_, idx.qp_range, Kokkos::ALL);
        InterpMatMul3(shape_interp, node_u, qp_u);

        // Interpolate rotation displacement
        auto node_r = Kokkos::subview(node_u_, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_r = Kokkos::subview(qp_r_, idx.qp_range, Kokkos::ALL);
        InterpQuaternion(shape_interp, node_r, qp_r);

        // Interpolate translation velocity
        auto node_u_dot = Kokkos::subview(node_u_dot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_u_dot = Kokkos::subview(qp_u_dot_, idx.qp_range, Kokkos::ALL);
        InterpMatMul3(shape_interp, node_u_dot, qp_u_dot);

        // Interpolate angular velocity
        auto node_omega = Kokkos::subview(node_u_dot_, idx.node_range, Kokkos::make_pair(3, 6));
        auto qp_omega = Kokkos::subview(qp_omega_, idx.qp_range, Kokkos::ALL);
        InterpMatMul3(shape_interp, node_omega, qp_omega);

        // Interpolate translation acceleration
        auto node_u_ddot = Kokkos::subview(node_u_ddot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_u_ddot = Kokkos::subview(qp_u_ddot_, idx.qp_range, Kokkos::ALL);
        InterpMatMul3(shape_interp, node_u_ddot, qp_u_ddot);

        // Interpolate angular acceleration
        auto node_omega_dot = Kokkos::subview(node_u_ddot_, idx.node_range, Kokkos::make_pair(3, 6));
        auto qp_omega_dot = Kokkos::subview(qp_omega_dot_, idx.qp_range, Kokkos::ALL);
        InterpMatMul3(shape_interp, node_omega_dot, qp_omega_dot);
    }
};

struct InterpolateQPRotation {
    Kokkos::View<BeamElemIndices*> elem_indices;  // Element indices
    oturb::View_NxN shape_interp_;                // Num Nodes x Num Quadrature points
    oturb::View_Nx7 node_pos_rot_;                // Node global position vector
    oturb::View_Nx4 qp_rot_;                      // quadrature point rotation

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        // Element specific views
        auto shape_interp = Kokkos::subview(
            shape_interp_, elem_indices[i_elem].node_range,
            Kokkos::make_pair(0, elem_indices[i_elem].num_qps)
        );
        auto node_rot =
            Kokkos::subview(node_pos_rot_, elem_indices[i_elem].node_range, Kokkos::make_pair(3, 7));
        auto qp_rot = Kokkos::subview(qp_rot_, elem_indices[i_elem].qp_range, Kokkos::ALL);

        InterpMatMul4(shape_interp, node_rot, qp_rot);

        // Normalize quaternions (rows)
        for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
            auto length = Kokkos::sqrt(
                Kokkos::pow(qp_rot(j, 0), 2) + Kokkos::pow(qp_rot(j, 1), 2) +
                Kokkos::pow(qp_rot(j, 2), 2) + Kokkos::pow(qp_rot(j, 3), 2)
            );
            if (length == 0.) {
                qp_rot(j, 0) = 1.;
                qp_rot(j, 3) = 0.;
                qp_rot(j, 2) = 0.;
                qp_rot(j, 1) = 0.;
            } else {
                qp_rot(j, 0) /= length;
                qp_rot(j, 3) /= length;
                qp_rot(j, 2) /= length;
                qp_rot(j, 1) /= length;
            }
        }
    }
};

struct InterpolateQPAngularVelocityAcceleration {
    Kokkos::View<BeamElemIndices*> elem_indices;  // Element indices
    oturb::View_NxN shape_interp_;                // Num Nodes x Num Quadrature points
    oturb::View_Nx6 node_trans_ang_;              // Node translation & angular velocity/acceleration
    oturb::View_Nx3 qp_ang_;                      // quadrature point angular velocity/acceleration

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        // Element specific views
        auto shape_interp = Kokkos::subview(
            shape_interp_, elem_indices[i_elem].node_range,
            Kokkos::make_pair(0, elem_indices[i_elem].num_qps)
        );
        auto node_rot = Kokkos::subview(
            node_trans_ang_, elem_indices[i_elem].node_range, Kokkos::make_pair(3, 6)
        );
        auto qp_rot = Kokkos::subview(qp_ang_, elem_indices[i_elem].qp_range, Kokkos::ALL);

        // Initialize qp_rot
        Kokkos::deep_copy(qp_rot, 0.);

        // Perform matrix-matrix multiplication
        for (int i = 0; i < elem_indices[i_elem].num_nodes; ++i) {
            for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
                for (int k = 0; k < 3; ++k) {
                    qp_rot(j, k) += node_rot(i, k) * shape_interp(i, j);
                }
            }
        }
    }
};

struct InterpolateQPPositionDerivative {
    Kokkos::View<BeamElemIndices*> elem_indices;  // Element indices
    oturb::View_NxN shape_deriv_;                 // Num Nodes x Num Quadrature points
    oturb::View_N qp_jacobian_;                   // Jacobians
    oturb::View_Nx7 node_pos_rot_;                // Node global position/rotation vector
    oturb::View_Nx3 qp_pos_deriv_;                // quadrature point position derivative

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        // Element specific views
        oturb::View_NxN shape_deriv = Kokkos::subview(
            shape_deriv_, elem_indices[i_elem].node_range,
            Kokkos::make_pair(0, elem_indices[i_elem].num_qps)
        );
        auto qp_pos_deriv =
            Kokkos::subview(qp_pos_deriv_, elem_indices[i_elem].qp_range, Kokkos::ALL);
        auto node_pos =
            Kokkos::subview(node_pos_rot_, elem_indices[i_elem].node_range, Kokkos::make_pair(0, 3));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, elem_indices[i_elem].qp_range);

        // Initialize qp_pos_deriv
        Kokkos::deep_copy(qp_pos_deriv, 0.);

        // Perform matrix-matrix multiplication
        for (int i = 0; i < elem_indices[i_elem].num_nodes; ++i) {
            for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
                for (int k = 0; k < 3; ++k) {
                    qp_pos_deriv(j, k) += node_pos(i, k) * shape_deriv(i, j);
                }
            }
        }

        // Divide each row by jacobian
        for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
            qp_pos_deriv(j, 0) /= qp_jacobian(j);
            qp_pos_deriv(j, 1) /= qp_jacobian(j);
            qp_pos_deriv(j, 2) /= qp_jacobian(j);
        }
    }
};

struct InterpolateQPRotationDerivative {
    Kokkos::View<BeamElemIndices*> elem_indices;  // Element indices
    oturb::View_NxN shape_deriv_;                 // Num Nodes x Num Quadrature points
    oturb::View_N qp_jacobian_;                   // Jacobians
    oturb::View_Nx7 node_pos_rot_;                // Node global position/rotation vector
    oturb::View_Nx4 qp_rot_deriv_;                // quadrature point rotation derivative

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        // Element specific views
        auto shape_deriv = Kokkos::subview(
            shape_deriv_, elem_indices[i_elem].node_range,
            Kokkos::make_pair(0, elem_indices[i_elem].num_qps)
        );
        auto qp_rot_deriv =
            Kokkos::subview(qp_rot_deriv_, elem_indices[i_elem].qp_range, Kokkos::ALL);
        auto node_rot =
            Kokkos::subview(node_pos_rot_, elem_indices[i_elem].node_range, Kokkos::make_pair(3, 7));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, elem_indices[i_elem].qp_range);

        // Initialize qp_rot_deriv
        Kokkos::deep_copy(qp_rot_deriv, 0.);

        // Perform matrix-matrix multiplication
        for (int i = 0; i < elem_indices[i_elem].num_nodes; ++i) {
            for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
                for (int k = 0; k < 4; ++k) {
                    qp_rot_deriv(j, k) += node_rot(i, k) * shape_deriv(i, j);
                }
            }
        }

        // Divide each row by jacobian
        for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
            qp_rot_deriv(j, 0) /= qp_jacobian(j);
            qp_rot_deriv(j, 1) /= qp_jacobian(j);
            qp_rot_deriv(j, 2) /= qp_jacobian(j);
        }
    }
};

struct CalculateJacobian {
    Kokkos::View<BeamElemIndices*> elem_indices;  // Element indices
    oturb::View_NxN shape_deriv_;                 // Num Nodes x Num Quadrature points
    oturb::View_Nx7 node_pos_rot_;                // Node global position/rotation vector
    oturb::View_Nx3 qp_pos_deriv_;                // quadrature point position derivative
    oturb::View_N qp_jacobian_;                   // Jacobians

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        // Element specific views
        oturb::View_NxN shape_deriv = Kokkos::subview(
            shape_deriv_, elem_indices[i_elem].node_range,
            Kokkos::make_pair(0, elem_indices[i_elem].num_qps)
        );
        auto qp_pos_deriv =
            Kokkos::subview(qp_pos_deriv_, elem_indices[i_elem].qp_range, Kokkos::ALL);
        auto node_pos =
            Kokkos::subview(node_pos_rot_, elem_indices[i_elem].node_range, Kokkos::make_pair(0, 3));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, elem_indices[i_elem].qp_range);

        // Initialize qp_pos_deriv
        Kokkos::deep_copy(qp_pos_deriv, 0.);

        // Perform matrix-matrix multiplication
        for (int i = 0; i < elem_indices[i_elem].num_nodes; ++i) {
            for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
                for (int k = 0; k < 3; ++k) {
                    qp_pos_deriv(j, k) += node_pos(i, k) * shape_deriv(i, j);
                }
            }
        }

        //  Loop through quadrature points
        for (int j = 0; j < elem_indices[i_elem].num_qps; ++j) {
            // Calculate Jacobian as norm of derivative
            qp_jacobian(j) = Kokkos::sqrt(
                Kokkos::pow(qp_pos_deriv(j, 0), 2.) + Kokkos::pow(qp_pos_deriv(j, 1), 2.) +
                Kokkos::pow(qp_pos_deriv(j, 2), 2.)
            );

            // Apply Jacobian to row
            qp_pos_deriv(j, 0) /= qp_jacobian(j);
            qp_pos_deriv(j, 1) /= qp_jacobian(j);
            qp_pos_deriv(j, 2) /= qp_jacobian(j);
        }
    }
};

//------------------------------------------------------------------------------
// Beams data structure
//------------------------------------------------------------------------------

struct Beams {
    size_t num;  // Number of beams

    Kokkos::View<BeamElemIndices*> elem_indices;  // View of element node and qp indices into views
    Kokkos::View<int*> node_state_indices;        // State row index for each node

    // Node-based data
    oturb::View_Nx7 node_x0;              // Inital position/rotation
    oturb::View_Nx7 node_u;               // State: translation/rotation displacement
    oturb::View_Nx6 node_u_dot;           // State: translation/rotation velocity
    oturb::View_Nx6 node_u_ddot;          // State: translation/rotation acceleration
    oturb::View_Nx6 node_force_app;       // Applied forces
    oturb::View_Nx6 node_force_elastic;   // Elastic forces
    oturb::View_Nx6 node_force_gravity;   // Gravity forces
    oturb::View_Nx6 node_force_inertial;  // Inertial forces
    oturb::View_Nx6 node_force_external;  // External forces
    oturb::View_Nx6 node_residual;        // Residual forces
    oturb::View_NxN node_iteration;       // Iteration matrix

    // Quadrature point data
    oturb::View_N qp_weight;       //
    oturb::View_N qp_jacobian;     //
    oturb::View_Nx3 qp_x0;         // initial position
    oturb::View_Nx3 qp_x0_prime;   // initial position derivative
    oturb::View_Nx4 qp_r0;         // initial rotation
    oturb::View_Nx6x6 qp_Mstar;    // Mass matrix in reference
    oturb::View_Nx6x6 qp_Cstar;    // Stiffness matrix in reference
    oturb::View_Nx3 qp_u;          // current translation displacement
    oturb::View_Nx3 qp_u_prime;    // current translation displacement derivative
    oturb::View_Nx3 qp_u_dot;      // current translation velocity
    oturb::View_Nx3 qp_u_ddot;     // current translation acceleration
    oturb::View_Nx4 qp_r;          // current rotation
    oturb::View_Nx4 qp_r_prime;    // current rotation derivative
    oturb::View_Nx3 qp_omega;      // current angular velocity
    oturb::View_Nx3 qp_omega_dot;  // current position/rotation
    oturb::View_Nx6 qp_strain;     // current translation/angular velocity
    oturb::View_Nx6 qp_Fc;         //
    oturb::View_Nx6 qp_Fd;         //
    oturb::View_Nx6 qp_Fi;         //
    oturb::View_Nx6 qp_Fg;         //
    oturb::View_Nx6 qp_Fext;       //
    oturb::View_Nx6x6 qp_RR0;      //
    oturb::View_Nx6x6 qp_Muu;      //
    oturb::View_Nx6x6 qp_Cuu;      //
    oturb::View_Nx6x6 qp_Ouu;      //
    oturb::View_Nx6x6 qp_Puu;      //
    oturb::View_Nx6x6 qp_Quu;      //
    oturb::View_Nx6x6 qp_Guu;      //
    oturb::View_Nx6x6 qp_Kuu;      //

    oturb::View_NxN shape_interp;  // shape function matrix for interpolation [Nodes x QPs]
    oturb::View_NxN shape_deriv;   // shape function matrix for derivative interp [Nodes x QPs]

    Beams(const size_t NumBeams, const size_t NumNodes, const size_t NumQPs, const size_t MaxElemQPs)
        : num(NumBeams),
          // Element Data
          elem_indices("elem_indices", NumBeams),
          node_state_indices("node_state_indices", NumNodes),
          // Node Data
          node_x0("node_x0", NumNodes),
          node_u("node_u", NumNodes),
          node_u_dot("node_u_dot", NumNodes),
          node_u_ddot("node_u_ddot", NumNodes),
          node_force_app("node_force_app", NumNodes),
          node_force_elastic("node_force_elastic", NumNodes),
          node_force_gravity("node_force_gravity", NumNodes),
          node_force_inertial("node_force_inertial", NumNodes),
          node_force_external("node_force_external", NumNodes),
          node_residual("node_residual", NumNodes),
          node_iteration("node_iteration", NumNodes, NumNodes),
          // Quadrature Point data
          qp_weight("qp_weight", NumQPs),
          qp_jacobian("qp_jacobian", NumQPs),
          qp_x0("qp_x0", NumQPs),
          qp_x0_prime("qp_x0_prime", NumQPs),
          qp_r0("qp_r0", NumQPs),
          qp_Mstar("qp_Mstar", NumQPs),
          qp_Cstar("qp_Cstar", NumQPs),
          qp_u("qp_u", NumQPs),
          qp_u_prime("qp_u_prime", NumQPs),
          qp_u_dot("qp_u_dot", NumQPs),
          qp_u_ddot("qp_u_ddot", NumQPs),
          qp_r("qp_r", NumQPs),
          qp_r_prime("qp_r_prime", NumQPs),
          qp_omega("qp_omega", NumQPs),
          qp_omega_dot("qp_omega_dot", NumQPs),
          qp_strain("qp_strain", NumQPs),
          qp_Fc("qp_Fc", NumQPs),
          qp_Fd("qp_Fd", NumQPs),
          qp_Fi("qp_Fi", NumQPs),
          qp_Fg("qp_Fg", NumQPs),
          qp_Fext("qp_Fext", NumQPs),
          qp_RR0("qp_RR0", NumQPs),
          qp_Muu("qp_Muu", NumQPs),
          qp_Cuu("qp_Cuu", NumQPs),
          qp_Ouu("qp_Ouu", NumQPs),
          qp_Puu("qp_Puu", NumQPs),
          qp_Quu("qp_Quu", NumQPs),
          qp_Guu("qp_Guu", NumQPs),
          qp_Kuu("qp_Kuu", NumQPs),
          shape_interp("shape_interp", NumNodes, MaxElemQPs),
          shape_deriv("deriv_interp", NumNodes, MaxElemQPs) {}

    // Update node states (displacement, velocity, acceleration)

    void UpdateState(View_Nx7 Q, View_Nx6 V, View_Nx6 A) {
        // Update node displacement, velocity, and acceleration
        Kokkos::parallel_for(
            "UpdateNodeState", this->node_u.extent(0),
            KOKKOS_LAMBDA(size_t i) {
                auto j = this->node_state_indices(i);
                for (size_t k = 0; k < this->node_u.extent(1); k++) {
                    this->node_u(i, k) = Q(j, k);
                }
                for (size_t k = 0; k < this->node_u_dot.extent(1); k++) {
                    this->node_u_dot(i, k) = V(j, k);
                }
                for (size_t k = 0; k < this->node_u_ddot.extent(1); k++) {
                    this->node_u_ddot(i, k) = A(j, k);
                }
            }
        );

        // Interpolate node positions to quadrature points
        Kokkos::parallel_for(
            "InterpolateQPPosition", this->num,
            InterpolateQPPosition{this->elem_indices, this->shape_interp, this->node_u, this->qp_u}
        );

        // Interpolate node rotations to quadrature points
        Kokkos::parallel_for(
            "InterpolateQPRotation", this->num,
            InterpolateQPRotation{this->elem_indices, this->shape_interp, this->node_u, this->qp_r}
        );

        // Interpolate node rotations to quadrature points
        Kokkos::parallel_for(
            "InterpolateQPRotationDerivative", this->num,
            InterpolateQPRotationDerivative{
                this->elem_indices, this->shape_interp, this->qp_jacobian, this->node_u,
                this->qp_r_prime}
        );
    }
};

// Initialize structure from array of beam element inputs
Beams InitializeBeams(std::vector<BeamInput> elem_inputs);

}  // namespace oturb