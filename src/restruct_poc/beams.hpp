#pragma once

#include <array>
#include <numeric>

#include <KokkosBlas.hpp>

#include "math_util.hpp"
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
    double s_;
    std::array<double, 7> x_;
    std::array<double, 7> q_;
    std::array<double, 6> v_;
    std::array<double, 6> a_;
    BeamNode(std::array<double, 7> x) : s_(0.), x_(std::move(x)), q_({0.}), v_({0.}), a_({0.}) {}
    BeamNode(
        std::array<double, 7> x, std::array<double, 7> q, std::array<double, 6> v,
        std::array<double, 6> a
    )
        : s_(0.), x_(std::move(x)), q_(std::move(q)), v_(std::move(v)), a_(std::move(a)) {}
    BeamNode(
        double s, std::array<double, 7> x, std::array<double, 7> q, std::array<double, 6> v,
        std::array<double, 6> a
    )
        : s_(s), x_(std::move(x)), q_(std::move(q)), v_(std::move(v)), a_(std::move(a)) {}
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
        // If node positions already set, return
        if (nodes.back().s_ != 0.)
            return;

        // Calculate distances between nodes in element
        std::vector<double> node_distances({0.});
        for (size_t i = 1; i < this->nodes.size(); i++) {
            node_distances.push_back(sqrt(
                pow(this->nodes[i].x_[0] - this->nodes[i - 1].x_[0], 2) +
                pow(this->nodes[i].x_[1] - this->nodes[i - 1].x_[1], 2) +
                pow(this->nodes[i].x_[2] - this->nodes[i - 1].x_[2], 2)
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
            this->nodes[i].s_ = node_cumulative_distances[i] / length;
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

        InterpQuaternion(shape_interp, node_rot, qp_rot);
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

        InterpVector4Deriv(shape_deriv, qp_jacobian, node_rot, qp_rot_deriv);
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
        auto idx = elem_indices[i_elem];
        // Element specific views
        oturb::View_NxN shape_deriv =
            Kokkos::subview(shape_deriv_, idx.node_range, Kokkos::make_pair(0, idx.num_qps));
        auto qp_pos_deriv = Kokkos::subview(qp_pos_deriv_, idx.qp_range, Kokkos::ALL);
        auto node_pos = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);

        // Interpolate quadrature point position derivative from node position
        InterpVector3(shape_deriv, node_pos, qp_pos_deriv);

        //  Loop through quadrature points
        for (int j = 0; j < idx.num_qps; ++j) {
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

struct InterpolateQPState {
    Kokkos::View<BeamElemIndices*> elem_indices;  // Element indices

    oturb::View_NxN shape_interp_;  // Num Nodes x Num Quadrature points
    oturb::View_NxN shape_deriv_;   // Num Nodes x Num Quadrature points
    oturb::View_N qp_jacobian_;     // Num Nodes x Num Quadrature points
    oturb::View_Nx7 node_u_;        // Node translation & rotation displacement
    oturb::View_Nx6 node_u_dot_;    // Node translation & angular velocity
    oturb::View_Nx6 node_u_ddot_;   // Node translation & angular acceleration

    oturb::View_Nx3 qp_u_;          // qp translation displacement
    oturb::View_Nx3 qp_u_prime_;    // qp translation displacement derivative
    oturb::View_Nx4 qp_r_;          // qp rotation displacement
    oturb::View_Nx4 qp_r_prime_;    // qp rotation displacement derivative
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
        auto node_u = Kokkos::subview(node_u_, idx.node_range, Kokkos::make_pair(0, 3));
        auto node_r = Kokkos::subview(node_u_, idx.node_range, Kokkos::make_pair(3, 7));

        // Interpolate translation displacement
        auto qp_u = Kokkos::subview(qp_u_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_u, qp_u);

        // Interpolate translation displacement derivative
        auto qp_u_prime = Kokkos::subview(qp_u_prime_, idx.qp_range, Kokkos::ALL);
        InterpVector3Deriv(shape_deriv, qp_jacobian, node_u, qp_u_prime);

        // Interpolate rotation displacement
        auto qp_r = Kokkos::subview(qp_r_, idx.qp_range, Kokkos::ALL);
        InterpQuaternion(shape_interp, node_r, qp_r);

        // Interpolate rotation displacement derivative
        auto qp_r_prime = Kokkos::subview(qp_r_prime_, idx.qp_range, Kokkos::ALL);
        InterpVector4Deriv(shape_deriv, qp_jacobian, node_r, qp_r_prime);

        // Interpolate translation velocity
        auto node_u_dot = Kokkos::subview(node_u_dot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_u_dot = Kokkos::subview(qp_u_dot_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_u_dot, qp_u_dot);

        // Interpolate angular velocity
        auto node_omega = Kokkos::subview(node_u_dot_, idx.node_range, Kokkos::make_pair(3, 6));
        auto qp_omega = Kokkos::subview(qp_omega_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_omega, qp_omega);

        // Interpolate translation acceleration
        auto node_u_ddot = Kokkos::subview(node_u_ddot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_u_ddot = Kokkos::subview(qp_u_ddot_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_u_ddot, qp_u_ddot);

        // Interpolate angular acceleration
        auto node_omega_dot = Kokkos::subview(node_u_ddot_, idx.node_range, Kokkos::make_pair(3, 6));
        auto qp_omega_dot = Kokkos::subview(qp_omega_dot_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_omega_dot, qp_omega_dot);
    }
};

struct CalculateRR0 {
    oturb::View_Nx4 qp_r0_;     // quadrature point initial rotation
    oturb::View_Nx4 qp_r_;      // quadrature rotation displacement
    oturb::View_Nx4 qRR0_;      // quaternion composition of RR0
    oturb::View_Nx6x6 qp_RR0_;  // quadrature global rotation

    KOKKOS_FUNCTION void operator()(const size_t i_qp) const {
        auto qR = Kokkos::subview(qp_r_, i_qp, Kokkos::ALL);
        auto qR0 = Kokkos::subview(qp_r0_, i_qp, Kokkos::ALL);
        auto qRR0 = Kokkos::subview(qRR0_, i_qp, Kokkos::ALL);
        auto RR0_11 =
            Kokkos::subview(qp_RR0_, i_qp, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        auto RR0_22 =
            Kokkos::subview(qp_RR0_, i_qp, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        QuaternionCompose(qR, qR0, qRR0);
        QuaternionToRotationMatrix(qRR0, RR0_11);
        Kokkos::deep_copy(RR0_22, RR0_11);
    }
};

struct CalculateMuu {
    oturb::View_Nx6x6 qp_RR0_;    //
    oturb::View_Nx6x6 qp_Mstar_;  //
    oturb::View_Nx6x6 qp_Muu_;    //
    oturb::View_Nx6x6 qp_Mtmp_;   //

    KOKKOS_FUNCTION
    void operator()(const size_t i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mstar = Kokkos::subview(qp_Mstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mtmp = Kokkos::subview(qp_Mtmp_, i_qp, Kokkos::ALL, Kokkos::ALL);
        MatMulAB(RR0, Mstar, Mtmp);
        MatMulABT(Mtmp, RR0, Muu);
    }
};

struct CalculateCuu {
    oturb::View_Nx6x6 qp_RR0_;    //
    oturb::View_Nx6x6 qp_Cstar_;  //
    oturb::View_Nx6x6 qp_Cuu_;    //
    oturb::View_Nx6x6 qp_Ctmp_;   //

    KOKKOS_FUNCTION
    void operator()(const size_t i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cstar = Kokkos::subview(qp_Cstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Ctmp = Kokkos::subview(qp_Ctmp_, i_qp, Kokkos::ALL, Kokkos::ALL);
        MatMulAB(RR0, Cstar, Ctmp);
        MatMulABT(Ctmp, RR0, Cuu);
    }
};

struct CalculateStrain {
    oturb::View_Nx3 qp_x0_prime_;  //
    oturb::View_Nx3 qp_u_prime_;   //
    oturb::View_Nx4 qp_r_;         //
    oturb::View_Nx4 qp_r_prime_;   //
    oturb::View_Nx3x4 qp_E_;       //
    oturb::View_Nx3 qp_V_;         //
    oturb::View_Nx6 qp_strain_;    //

    KOKKOS_FUNCTION
    void operator()(const size_t i_qp) const {
        auto x0_prime = Kokkos::subview(qp_x0_prime_, i_qp, Kokkos::ALL);
        auto u_prime = Kokkos::subview(qp_u_prime_, i_qp, Kokkos::ALL);
        auto R = Kokkos::subview(qp_r_, i_qp, Kokkos::ALL);
        auto R_prime = Kokkos::subview(qp_r_prime_, i_qp, Kokkos::ALL);
        auto E = Kokkos::subview(qp_E_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto R_x0_prime = Kokkos::subview(qp_V_, i_qp, Kokkos::ALL);
        auto e1 = Kokkos::subview(qp_strain_, i_qp, Kokkos::make_pair(0, 3));
        auto e2 = Kokkos::subview(qp_strain_, i_qp, Kokkos::make_pair(3, 6));

        QuaternionRotateVector(R, x0_prime, R_x0_prime);
        QuaternionDerivative(R, E);
        MatVecMulAB(E, R_prime, e2);
        for (size_t i = 0; i < 3; i++) {
            e1(i) = x0_prime(i) + u_prime(i) - R_x0_prime(i);
            e2(i) *= 2.;
        }
    }
};

struct CalculateForces {
    oturb::View_3 gravity;               //
    oturb::View_Nx6x6 qp_Muu_;           //
    oturb::View_Nx6x6 qp_Cuu_;           //
    oturb::View_Nx3 qp_x0_prime_;        //
    oturb::View_Nx3 qp_u_prime_;         //
    oturb::View_Nx3 qp_u_ddot_;          //
    oturb::View_Nx3 qp_omega_;           //
    oturb::View_Nx3 qp_omega_dot_;       //
    oturb::View_Nx6 qp_strain_;          //
    oturb::View_Nx3x3 eta_tilde_;        //
    oturb::View_Nx3x3 omega_tilde_;      //
    oturb::View_Nx3x3 omega_dot_tilde_;  //
    oturb::View_Nx3x3 x0pupSS_;          //
    oturb::View_Nx3x3 M_tilde_;          //
    oturb::View_Nx3x3 N_tilde_;          //
    oturb::View_Nx3x3 rho_;              //
    oturb::View_Nx3 eta_;                //
    oturb::View_Nx3 v_;                  // temporary vector
    oturb::View_Nx3x3 m3_;               // temporary matrix
    oturb::View_Nx6 qp_FC_;              //
    oturb::View_Nx6 qp_FD_;              //
    oturb::View_Nx6 qp_FI_;              //
    oturb::View_Nx6 qp_FG_;              //
    oturb::View_Nx6x6 qp_Ouu_;           //
    oturb::View_Nx6x6 qp_Puu_;           //
    oturb::View_Nx6x6 qp_Quu_;           //
    oturb::View_Nx6x6 qp_Guu_;           //
    oturb::View_Nx6x6 qp_Kuu_;           //

    KOKKOS_FUNCTION
    void operator()(const size_t i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0_prime = Kokkos::subview(qp_x0_prime_, i_qp, Kokkos::ALL);
        auto u_prime = Kokkos::subview(qp_u_prime_, i_qp, Kokkos::ALL);
        auto u_ddot = Kokkos::subview(qp_u_ddot_, i_qp, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_qp, Kokkos::ALL);
        auto omega_dot = Kokkos::subview(qp_omega_dot_, i_qp, Kokkos::ALL);
        auto strain = Kokkos::subview(qp_strain_, i_qp, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_dot_tilde = Kokkos::subview(omega_dot_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M_tilde = Kokkos::subview(M_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto v = Kokkos::subview(v_, i_qp, Kokkos::ALL);
        auto m3 = Kokkos::subview(m3_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FC = Kokkos::subview(qp_FC_, i_qp, Kokkos::ALL);
        auto FD = Kokkos::subview(qp_FD_, i_qp, Kokkos::ALL);
        auto FI = Kokkos::subview(qp_FI_, i_qp, Kokkos::ALL);
        auto FG = Kokkos::subview(qp_FG_, i_qp, Kokkos::ALL);
        auto Ouu = Kokkos::subview(qp_Ouu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Puu = Kokkos::subview(qp_Puu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Quu = Kokkos::subview(qp_Quu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Guu = Kokkos::subview(qp_Guu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Kuu = Kokkos::subview(qp_Kuu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        auto C12 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        auto C21 = Kokkos::subview(Cuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));

        // Mass matrix components
        auto m = Muu(0, 0);
        if (m == 0.) {
            Kokkos::deep_copy(eta, 0.);
        } else {
            eta(0) = Muu(5, 1) / m;
            eta(1) = -Muu(5, 0) / m;
            eta(2) = Muu(4, 0) / m;
        }
        Kokkos::deep_copy(
            rho, Kokkos::subview(Muu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6))
        );
        VecTilde(eta, eta_tilde);

        // Temporary variable used in many other calcs
        for (size_t i = 0; i < 3; i++) {
            v(i) = x0_prime(i) + u_prime(i);
        }
        VecTilde(v, x0pupSS);

        // Elastic Force FC and it's components
        MatVecMulAB(Cuu, strain, FC);
        auto N = Kokkos::subview(FC, Kokkos::make_pair(0, 3));
        auto M = Kokkos::subview(FC, Kokkos::make_pair(3, 6));
        VecTilde(M, M_tilde);
        VecTilde(N, N_tilde);

        // Elastic Force FD and it's components
        Kokkos::deep_copy(FD, 0.);
        MatVecMulATB(x0pupSS, N, Kokkos::subview(FD, Kokkos::make_pair(3, 6)));

        // Inertial forces
        VecTilde(omega, omega_tilde);
        VecTilde(omega_dot, omega_dot_tilde);
        auto FI_1 = Kokkos::subview(FI, Kokkos::make_pair(0, 3));
        MatMulAB(omega_tilde, omega_tilde, m3);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                m3(i, j) += omega_dot_tilde(i, j);
                m3(i, j) *= m;
            }
        }
        MatVecMulAB(m3, eta, FI_1);
        for (size_t i = 0; i < 3; i++) {
            FI_1(i) += u_ddot(i) * m;
        }
        auto FI_2 = Kokkos::subview(FI, Kokkos::make_pair(3, 6));
        VecScale(u_ddot, m, v);
        MatVecMulAB(eta_tilde, v, FI_2);
        MatVecMulAB(rho, omega_dot, v);
        for (size_t i = 0; i < 3; i++) {
            FI_2(i) += v(i);
        }
        MatMulAB(omega_tilde, rho, m3);
        MatVecMulAB(m3, omega, v);
        for (size_t i = 0; i < 3; i++) {
            FI_2(i) += v(i);
        }

        // Gravity force
        VecScale(gravity, m, v);
        Kokkos::deep_copy(Kokkos::subview(FG, Kokkos::make_pair(0, 3)), v);
        MatVecMulAB(eta_tilde, v, Kokkos::subview(FG, Kokkos::make_pair(3, 6)));

        // Ouu
        Kokkos::deep_copy(Ouu, 0.);
        auto Ouu_12 = Kokkos::subview(Ouu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        auto Ouu_22 = Kokkos::subview(Ouu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(C11, x0pupSS, Ouu_12);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Ouu_12(i, j) -= N_tilde(i, j);
            }
        }
        MatMulAB(C21, x0pupSS, Ouu_22);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Ouu_22(i, j) -= M_tilde(i, j);
            }
        }

        // Puu
        Kokkos::deep_copy(Puu, 0.);
        auto Puu_21 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
        MatMulATB(x0pupSS, C11, Puu_21);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Puu_21(i, j) += N_tilde(i, j);
            }
        }
        auto Puu_22 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulATB(x0pupSS, C12, Puu_22);

        // Quu
        Kokkos::deep_copy(Quu, 0.);
        auto Quu_22 = Kokkos::subview(Quu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(C11, x0pupSS, m3);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                m3(i, j) -= N_tilde(i, j);
            }
        }
        MatMulATB(x0pupSS, m3, Quu_22);

        // Inertia gyroscopic matrix
        Kokkos::deep_copy(Guu, 0.);
        // self.Guu.fixed_view_mut::<3, 3>(0, 3).copy_from(
        //     &((omega.tilde() * m * eta).tilde().transpose()
        //         + omega.tilde() * m * eta.tilde().transpose()),
        // );
        // self.Guu
        //     .fixed_view_mut::<3, 3>(3, 3)
        //     .copy_from(&(omega.tilde() * rho - (rho * omega).tilde()));

        // Inertia stiffness matrix
        Kokkos::deep_copy(Kuu, 0.);
        // self.Kuu.fixed_view_mut::<3, 3>(0, 3).copy_from(
        //     &((omega_dot.tilde() + omega.tilde() * omega.tilde()) * m * eta.tilde().transpose()),
        // );
        // self.Kuu.fixed_view_mut::<3, 3>(3, 3).copy_from(
        //     &(u_ddot.tilde() * m * eta.tilde()
        //         + (rho * omega_dot.tilde() - (rho * omega_dot).tilde())
        //         + omega.tilde() * (rho * omega.tilde() - (rho * omega).tilde())),
        // );
    }
};

//------------------------------------------------------------------------------
// Beams data structure
//------------------------------------------------------------------------------

struct Beams {
    size_t num_beams_;  // Number of beams
    size_t num_nodes_;  // Number of nodes
    size_t num_qps_;    // Number of quadrature points

    Kokkos::View<BeamElemIndices*> elem_indices;  // View of element node and qp indices into views
    Kokkos::View<int*> node_state_indices;        // State row index for each node

    oturb::View_3 gravity;

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
    oturb::View_N qp_weight;       // Integration weights
    oturb::View_N qp_jacobian;     // Jacobian vector
    oturb::View_Nx6x6 qp_Mstar;    // Mass matrix in material frame
    oturb::View_Nx6x6 qp_Cstar;    // Stiffness matrix in material frame
    oturb::View_Nx3 qp_x0;         // Initial position
    oturb::View_Nx3 qp_x0_prime;   // Initial position derivative
    oturb::View_Nx4 qp_r0;         // Initial rotation
    oturb::View_Nx3 qp_u;          // State: translation displacement
    oturb::View_Nx3 qp_u_prime;    // State: translation displacement derivative
    oturb::View_Nx3 qp_u_dot;      // State: translation velocity
    oturb::View_Nx3 qp_u_ddot;     // State: translation acceleration
    oturb::View_Nx4 qp_r;          // State: rotation
    oturb::View_Nx4 qp_r_prime;    // State: rotation derivative
    oturb::View_Nx3 qp_omega;      // State: angular velocity
    oturb::View_Nx3 qp_omega_dot;  // State: position/rotation
    oturb::View_Nx6 qp_strain;     //
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

    // Scratch variables to be replaced later
    oturb::View_Nx6x6 M_6x6;   //
    oturb::View_Nx3x4 M_3x4;   //
    oturb::View_Nx3x3 M1_3x3;  //
    oturb::View_Nx3x3 M2_3x3;  //
    oturb::View_Nx3x3 M3_3x3;  //
    oturb::View_Nx3x3 M4_3x3;  //
    oturb::View_Nx3x3 M5_3x3;  //
    oturb::View_Nx3x3 M6_3x3;  //
    oturb::View_Nx3x3 M7_3x3;  //
    oturb::View_Nx3x3 M8_3x3;  //
    oturb::View_Nx3 V1_3;      //
    oturb::View_Nx3 V2_3;      //
    oturb::View_Nx3 V3_3;      //
    oturb::View_Nx4 qp_quat;   //

    Beams(
        const size_t num_beams, const size_t num_nodes, const size_t num_qps,
        const size_t max_elem_qps
    )
        : num_beams_(num_beams),
          num_nodes_(num_nodes),
          num_qps_(num_qps),
          // Element Data
          elem_indices("elem_indices", num_beams),
          node_state_indices("node_state_indices", num_nodes),
          gravity("gravity"),
          // Node Data
          node_x0("node_x0", num_nodes),
          node_u("node_u", num_nodes),
          node_u_dot("node_u_dot", num_nodes),
          node_u_ddot("node_u_ddot", num_nodes),
          node_force_app("node_force_app", num_nodes),
          node_force_elastic("node_force_elastic", num_nodes),
          node_force_gravity("node_force_gravity", num_nodes),
          node_force_inertial("node_force_inertial", num_nodes),
          node_force_external("node_force_external", num_nodes),
          node_residual("node_residual", num_nodes),
          node_iteration("node_iteration", num_nodes, num_nodes),
          // Quadrature Point data
          qp_weight("qp_weight", num_qps),
          qp_jacobian("qp_jacobian", num_qps),
          qp_Mstar("qp_Mstar", num_qps),
          qp_Cstar("qp_Cstar", num_qps),
          qp_x0("qp_x0", num_qps),
          qp_x0_prime("qp_x0_prime", num_qps),
          qp_r0("qp_r0", num_qps),
          qp_u("qp_u", num_qps),
          qp_u_prime("qp_u_prime", num_qps),
          qp_u_dot("qp_u_dot", num_qps),
          qp_u_ddot("qp_u_ddot", num_qps),
          qp_r("qp_r", num_qps),
          qp_r_prime("qp_r_prime", num_qps),
          qp_omega("qp_omega", num_qps),
          qp_omega_dot("qp_omega_dot", num_qps),
          qp_strain("qp_strain", num_qps),
          qp_Fc("qp_Fc", num_qps),
          qp_Fd("qp_Fd", num_qps),
          qp_Fi("qp_Fi", num_qps),
          qp_Fg("qp_Fg", num_qps),
          qp_Fext("qp_Fext", num_qps),
          qp_RR0("qp_RR0", num_qps),
          qp_Muu("qp_Muu", num_qps),
          qp_Cuu("qp_Cuu", num_qps),
          qp_Ouu("qp_Ouu", num_qps),
          qp_Puu("qp_Puu", num_qps),
          qp_Quu("qp_Quu", num_qps),
          qp_Guu("qp_Guu", num_qps),
          qp_Kuu("qp_Kuu", num_qps),
          shape_interp("shape_interp", num_nodes, max_elem_qps),
          shape_deriv("deriv_interp", num_nodes, max_elem_qps),
          // Scratch
          M_6x6("M1_6x6", num_qps),
          M_3x4("M_3x4", num_qps),
          M1_3x3("R1_3x3", num_qps),
          M2_3x3("R1_3x3", num_qps),
          M3_3x3("R1_3x3", num_qps),
          M4_3x3("R1_3x3", num_qps),
          M5_3x3("R1_3x3", num_qps),
          M6_3x3("R1_3x3", num_qps),
          M7_3x3("R1_3x3", num_qps),
          M8_3x3("R1_3x3", num_qps),
          V1_3("V_3", num_qps),
          V2_3("V_3", num_qps),
          V3_3("V_3", num_qps),
          qp_quat("Quat_4", num_qps) {}

    // Update node states (displacement, velocity, acceleration) and interpolate to quadrature points
    void UpdateState(View_Nx7 Q, View_Nx6 V, View_Nx6 A) {
        // Copy displacement, velocity, and acceleration to nodes
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

        // Interpolate node state to quadrature points
        Kokkos::parallel_for(
            "InterpolateQPState", this->num_beams_,
            InterpolateQPState{
                this->elem_indices, this->shape_interp, this->shape_deriv, this->qp_jacobian,
                this->node_u, this->node_u_dot, this->node_u_ddot, this->qp_u, this->qp_u_prime,
                this->qp_r, this->qp_r_prime, this->qp_u_dot, this->qp_omega, this->qp_u_ddot,
                this->qp_omega_dot}
        );
    }

    void CalculateQPData() {
        // Calculate RR0 matrix
        Kokkos::parallel_for(
            "CalculateRR0", this->num_qps_,
            CalculateRR0{
                this->qp_r0,
                this->qp_r,
                this->qp_quat,
                this->qp_RR0,
            }
        );

        // Calculate Muu matrix
        Kokkos::parallel_for(
            "CalculateMuu", this->num_qps_,
            CalculateMuu{this->qp_RR0, this->qp_Mstar, this->qp_Muu, this->M_6x6}
        );

        // Calculate Cuu matrix
        Kokkos::parallel_for(
            "CalculateCuu", this->num_qps_,
            CalculateCuu{this->qp_RR0, this->qp_Cstar, this->qp_Cuu, this->M_6x6}
        );

        // Calculate strain
        Kokkos::parallel_for(
            "CalculateStrain", this->num_qps_,
            CalculateStrain{
                this->qp_x0_prime,
                this->qp_u_prime,
                this->qp_r,
                this->qp_r_prime,
                this->M_3x4,
                this->V1_3,
                this->qp_strain,
            }
        );

        // Calculate Forces
        Kokkos::parallel_for(
            "CalculateForces", this->num_qps_,
            CalculateForces{
                this->gravity,    this->qp_Muu,    this->qp_Cuu,   this->qp_x0_prime,
                this->qp_u_prime, this->qp_u_ddot, this->qp_omega, this->qp_omega_dot,
                this->qp_strain,  this->M1_3x3,    this->M2_3x3,   this->M3_3x3,
                this->M4_3x3,     this->M5_3x3,    this->M6_3x3,   this->M7_3x3,
                this->V1_3,       this->V2_3,      this->M8_3x3,   this->qp_Fc,
                this->qp_Fd,      this->qp_Fi,     this->qp_Fg,    this->qp_Ouu,
                this->qp_Puu,     this->qp_Quu,    this->qp_Guu,   this->qp_Kuu,
            }
        );
    }
};

// Initialize structure from array of beam element inputs
Beams InitializeBeams(std::vector<BeamInput> elem_inputs);

}  // namespace oturb