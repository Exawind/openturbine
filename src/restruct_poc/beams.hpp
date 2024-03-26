#pragma once

#include <array>
#include <numeric>

#include <KokkosBlas.hpp>

#include "beams_calcs.hpp"
#include "interpolation.h"
#include "types.hpp"

#include "src/gebt_poc/quadrature.h"

namespace oturb {

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

struct BeamsInit {
    std::array<double, 3> gravity_;
    std::vector<BeamInput> inputs_;

    size_t NumBeams() const { return inputs_.size(); };
    size_t NumNodes() const {
        size_t num_nodes = 0;
        for (const auto& input : this->inputs_) {
            num_nodes += input.nodes.size();
        }
        return num_nodes;
    }
    size_t NumQuadraturePoints() const {
        size_t num_qps = 0;
        for (const auto& input : this->inputs_) {
            num_qps += input.quadrature.GetNumberOfQuadraturePoints();
        }
        return num_qps;
    }
    size_t MaxElemQuadraturePoints() const {
        size_t max_elem_qps = 0;
        for (const auto& input : this->inputs_) {
            max_elem_qps = std::max(max_elem_qps, input.quadrature.GetNumberOfQuadraturePoints());
        }
        return max_elem_qps;
    }
};

//------------------------------------------------------------------------------
// Element initialization
//------------------------------------------------------------------------------

void PopulateElementViews(
    const BeamInput& input, View_Nx7 node_x0, View_Nx7 node_u, View_Nx6 node_u_dot,
    View_Nx6 node_u_ddot, View_N qp_weight, View_Nx6x6 qp_Mstar, View_Nx6x6 qp_Cstar,
    View_NxN shape_interp, View_NxN shape_deriv
) {
    //--------------------------------------------------------------------------
    // Calculate element's node and quadrature point positions [-1,1]
    //--------------------------------------------------------------------------

    std::vector<double> node_xi(input.nodes.size());

    // Loop through nodes and convert 's' [0,1] position to xi
    for (size_t i = 0; i < input.nodes.size(); ++i) {
        node_xi[i] = 2 * input.nodes[i].s_ - 1;
    }

    //--------------------------------------------------------------------------
    // Populate node data
    //--------------------------------------------------------------------------

    // Loop through nodes
    for (size_t j = 0; j < input.nodes.size(); ++j) {
        // Transfer initial position
        for (size_t k = 0; k < input.nodes[j].x_.size(); ++k) {
            node_x0(j, k) = input.nodes[j].x_[k];
        }

        // Transfer initial displacement
        for (size_t k = 0; k < input.nodes[j].q_.size(); ++k) {
            node_u(j, k) = input.nodes[j].q_[k];
        }

        // Transfer initial velocity
        for (size_t k = 0; k < input.nodes[j].v_.size(); ++k) {
            node_u_dot(j, k) = input.nodes[j].v_[k];
        }

        // Transfer initial acceleration
        for (size_t k = 0; k < input.nodes[j].a_.size(); ++k) {
            node_u_ddot(j, k) = input.nodes[j].a_[k];
        }
    }

    //--------------------------------------------------------------------------
    // Populate quadrature point weights
    //--------------------------------------------------------------------------

    // Get vector of quadrature weights
    auto qp_w = input.quadrature.GetQuadratureWeights();

    for (size_t j = 0; j < qp_w.size(); ++j) {
        qp_weight(j) = qp_w[j];
    }

    //--------------------------------------------------------------------------
    // Populate shape function interpolation/derivative matrices
    //--------------------------------------------------------------------------

    std::vector<double> weights;

    // Get vector of quadrature points
    auto qp_xi = input.quadrature.GetQuadraturePoints();

    // Loop through quadrature points
    for (size_t j = 0; j < qp_xi.size(); ++j) {
        // Get interpolation weights to go from nodes to this QP
        LagrangePolynomialInterpWeights(qp_xi[j], node_xi, weights);

        // Copy interp weights to host matrix
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_interp(k, j) = weights[k];
        }

        // Get derivative weights to go from nodes to this QP
        LagrangePolynomialDerivWeights(qp_xi[j], node_xi, weights);

        // Copy deriv weights to host matrix
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_deriv(k, j) = weights[k];
        }
    }

    //--------------------------------------------------------------------------
    // Interpolate section mass and stiffness matrices to quadrature points
    // TODO: remove assumption that s runs from 0 to 1.
    //--------------------------------------------------------------------------

    // Get section positions [-1,1]
    std::vector<double> section_xi(input.sections.size());

    // Loop through sections and convert 's' [0,1] position to xi
    for (size_t i = 0; i < input.sections.size(); ++i) {
        section_xi[i] = 2 * input.sections[i].s - 1;
    }

    // Fill view with zeros
    Kokkos::deep_copy(qp_Mstar, 0.);
    Kokkos::deep_copy(qp_Cstar, 0.);

    // Loop through quadrature points and calculate section weights for interp
    for (size_t i = 0; i < qp_xi.size(); ++i) {
        // Calculate weights
        LagrangePolynomialInterpWeights(qp_xi[i], section_xi, weights);

        // Loop through sections
        for (size_t j = 0; j < section_xi.size(); ++j) {
            for (size_t m = 0; m < 6; ++m) {
                for (size_t n = 0; n < 6; ++n) {
                    qp_Mstar(i, m, n) += input.sections[j].M_star[m][n] * weights[j];
                    qp_Cstar(i, m, n) += input.sections[j].C_star[m][n] * weights[j];
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Beams data structure
//------------------------------------------------------------------------------

struct Beams {
    size_t num_beams_;  // Number of beams
    size_t num_nodes_;  // Number of nodes
    size_t num_qps_;    // Number of quadrature points

    Kokkos::View<BeamElemIndices*> elem_indices;  // View of element node and qp indices into views
    Kokkos::View<size_t*> node_state_indices;     // State row index for each node

    View_3 gravity;

    // Node-based data
    View_Nx7 node_x0;         // Inital position/rotation
    View_Nx7 node_u;          // State: translation/rotation displacement
    View_Nx6 node_u_dot;      // State: translation/rotation velocity
    View_Nx6 node_u_ddot;     // State: translation/rotation acceleration
    View_Nx6 node_FE;         // Elastic forces
    View_Nx6 node_FG;         // Gravity forces
    View_Nx6 node_FI;         // Inertial forces
    View_Nx6 node_FX;         // External forces
    View_N node_residual;     // Residual forces
    View_NxN node_iteration;  // Iteration matrix

    // Quadrature point data
    View_N qp_weight;       // Integration weights
    View_N qp_jacobian;     // Jacobian vector
    View_Nx6x6 qp_Mstar;    // Mass matrix in material frame
    View_Nx6x6 qp_Cstar;    // Stiffness matrix in material frame
    View_Nx3 qp_x0;         // Initial position
    View_Nx3 qp_x0_prime;   // Initial position derivative
    View_Nx4 qp_r0;         // Initial rotation
    View_Nx3 qp_u;          // State: translation displacement
    View_Nx3 qp_u_prime;    // State: translation displacement derivative
    View_Nx3 qp_u_dot;      // State: translation velocity
    View_Nx3 qp_u_ddot;     // State: translation acceleration
    View_Nx4 qp_r;          // State: rotation
    View_Nx4 qp_r_prime;    // State: rotation derivative
    View_Nx3 qp_omega;      // State: angular velocity
    View_Nx3 qp_omega_dot;  // State: position/rotation
    View_Nx6 qp_strain;     //
    View_Nx6 qp_Fc;         //
    View_Nx6 qp_Fd;         //
    View_Nx6 qp_Fi;         //
    View_Nx6 qp_Fg;
    View_Nx6x6 qp_RR0;  //
    View_Nx6x6 qp_Muu;  //
    View_Nx6x6 qp_Cuu;  //
    View_Nx6x6 qp_Ouu;  //
    View_Nx6x6 qp_Puu;  //
    View_Nx6x6 qp_Quu;  //
    View_Nx6x6 qp_Guu;  //
    View_Nx6x6 qp_Kuu;  //

    View_NxN shape_interp;  // shape function matrix for interpolation [Nodes x QPs]
    View_NxN shape_deriv;   // shape function matrix for derivative interp [Nodes x QPs]

    // Scratch variables to be replaced later
    View_Nx6x6 M_6x6;   //
    View_Nx3x4 M_3x4;   //
    View_Nx3x3 M1_3x3;  //
    View_Nx3x3 M2_3x3;  //
    View_Nx3x3 M3_3x3;  //
    View_Nx3x3 M4_3x3;  //
    View_Nx3x3 M5_3x3;  //
    View_Nx3x3 M6_3x3;  //
    View_Nx3x3 M7_3x3;  //
    View_Nx3x3 M8_3x3;  //
    View_Nx3x3 M9_3x3;  //
    View_Nx3 V1_3;      //
    View_Nx3 V2_3;      //
    View_Nx3 V3_3;      //
    View_Nx4 qp_quat;   //

    Beams(const BeamsInit beams_init)
        : Beams(
              beams_init.NumBeams(), beams_init.NumNodes(), beams_init.NumQuadraturePoints(),
              beams_init.MaxElemQuadraturePoints()
          ) {
        //----------------------------------------------------------------------
        // Create host mirror of views for initialization from init data
        //----------------------------------------------------------------------

        auto host_gravity = Kokkos::create_mirror(this->gravity);

        auto host_elem_indices = Kokkos::create_mirror(this->elem_indices);
        auto host_node_x0 = Kokkos::create_mirror(this->node_x0);
        auto host_node_u = Kokkos::create_mirror(this->node_u);
        auto host_node_u_dot = Kokkos::create_mirror(this->node_u_dot);
        auto host_node_u_ddot = Kokkos::create_mirror(this->node_u_ddot);

        auto host_qp_weight = Kokkos::create_mirror(this->qp_weight);
        auto host_qp_Mstar = Kokkos::create_mirror(this->qp_Mstar);
        auto host_qp_Cstar = Kokkos::create_mirror(this->qp_Cstar);

        auto host_shape_interp = Kokkos::create_mirror(this->shape_interp);
        auto host_shape_deriv = Kokkos::create_mirror(this->shape_deriv);

        //----------------------------------------------------------------------
        // Set gravity
        //----------------------------------------------------------------------

        host_gravity(0) = beams_init.gravity_[0];
        host_gravity(1) = beams_init.gravity_[1];
        host_gravity(2) = beams_init.gravity_[2];

        //----------------------------------------------------------------------
        // Populate indices and element host views from input data
        //----------------------------------------------------------------------

        size_t node_counter = 0;
        size_t qp_counter = 0;

        // Loop through element inputs
        for (size_t i = 0; i < beams_init.inputs_.size(); i++) {
            // Calculate node indices for this element
            size_t num_nodes = beams_init.inputs_[i].nodes.size();
            auto node_range = Kokkos::make_pair(node_counter, node_counter + num_nodes);
            node_counter += num_nodes;

            // Calculate quadrature point indices for this element
            size_t num_qps = beams_init.inputs_[i].quadrature.GetNumberOfQuadraturePoints();
            auto qp_range = Kokkos::make_pair(qp_counter, qp_counter + num_qps);
            qp_counter += num_qps;

            // Populate element index
            host_elem_indices[i].num_nodes = num_nodes;
            host_elem_indices[i].node_range = node_range;
            host_elem_indices[i].num_qps = num_qps;
            host_elem_indices[i].qp_range = qp_range;

            // Populate views for this element
            PopulateElementViews(
                beams_init.inputs_[i],  // Element inputs
                Kokkos::subview(host_node_x0, node_range, Kokkos::ALL),
                Kokkos::subview(host_node_u, node_range, Kokkos::ALL),
                Kokkos::subview(host_node_u_dot, node_range, Kokkos::ALL),
                Kokkos::subview(host_node_u_ddot, node_range, Kokkos::ALL),
                Kokkos::subview(host_qp_weight, qp_range),
                Kokkos::subview(host_qp_Mstar, qp_range, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(host_qp_Cstar, qp_range, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(
                    host_shape_interp, node_range, Kokkos::make_pair((size_t)0, num_qps)
                ),
                Kokkos::subview(host_shape_deriv, node_range, Kokkos::make_pair((size_t)0, num_qps))
            );
        }

        //----------------------------------------------------------------------
        // Copy host data to beam views
        //----------------------------------------------------------------------

        // Copy from host to device
        Kokkos::deep_copy(this->gravity, host_gravity);
        Kokkos::deep_copy(this->elem_indices, host_elem_indices);
        Kokkos::deep_copy(this->node_x0, host_node_x0);
        Kokkos::deep_copy(this->node_u, host_node_u);
        Kokkos::deep_copy(this->node_u_dot, host_node_u_dot);
        Kokkos::deep_copy(this->node_u_ddot, host_node_u_ddot);
        Kokkos::deep_copy(this->qp_weight, host_qp_weight);
        Kokkos::deep_copy(this->qp_Mstar, host_qp_Mstar);
        Kokkos::deep_copy(this->qp_Cstar, host_qp_Cstar);
        Kokkos::deep_copy(this->shape_interp, host_shape_interp);
        Kokkos::deep_copy(this->shape_deriv, host_shape_deriv);

        //----------------------------------------------------------------------
        // Perform parallel initialization of data
        //----------------------------------------------------------------------

        // Set state index for each node
        // TODO: update for assembly where state may apply to multiple nodes in different elements
        Kokkos::parallel_for(
            "SetNodeStateIndices", this->node_u.extent(0),
            KOKKOS_LAMBDA(size_t i) { this->node_state_indices(i) = i; }
        );

        // Interpolate node positions to quadrature points
        Kokkos::parallel_for(
            "InterpolateQPPosition", this->num_beams_,
            InterpolateQPPosition{this->elem_indices, this->shape_interp, this->node_x0, this->qp_x0}
        );

        // Interpolate node rotations to quadrature points
        Kokkos::parallel_for(
            "InterpolateQPRotation", this->num_beams_,
            InterpolateQPRotation{this->elem_indices, this->shape_interp, this->node_x0, this->qp_r0}
        );

        // Calculate derivative of position (Jacobian) and x0_prime
        Kokkos::parallel_for(
            "CalculateJacobian", this->num_beams_,
            CalculateJacobian{
                this->elem_indices,
                this->shape_deriv,
                this->node_x0,
                this->qp_x0_prime,
                this->qp_jacobian,
            }
        );

        // Interpolate initial state to quadrature points
        Kokkos::parallel_for(
            "InterpolateQPState", this->num_beams_,
            InterpolateQPState{
                this->elem_indices, this->shape_interp, this->shape_deriv, this->qp_jacobian,
                this->node_u, this->node_u_dot, this->node_u_ddot, this->qp_u, this->qp_u_prime,
                this->qp_r, this->qp_r_prime, this->qp_u_dot, this->qp_omega, this->qp_u_ddot,
                this->qp_omega_dot}
        );
    }

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
          node_FE("node_force_elastic", num_nodes),
          node_FG("node_force_gravity", num_nodes),
          node_FI("node_force_inertial", num_nodes),
          node_FX("node_force_external", num_nodes),
          node_residual("node_residual", num_nodes * 6),
          node_iteration("node_iteration", num_nodes * 6, num_nodes * 6),
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
          M9_3x3("R1_3x3", num_qps),
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
            CalculateForcesAndMatrices{
                this->gravity,    this->qp_Muu,    this->qp_Cuu,   this->qp_x0_prime,
                this->qp_u_prime, this->qp_u_ddot, this->qp_omega, this->qp_omega_dot,
                this->qp_strain,  this->M1_3x3,    this->M2_3x3,   this->M3_3x3,
                this->M4_3x3,     this->M5_3x3,    this->M6_3x3,   this->M7_3x3,
                this->V1_3,       this->V2_3,      this->V3_3,     this->M8_3x3,
                this->M9_3x3,     this->qp_Fc,     this->qp_Fd,    this->qp_Fi,
                this->qp_Fg,      this->qp_Ouu,    this->qp_Puu,   this->qp_Quu,
                this->qp_Guu,     this->qp_Kuu,
            }
        );
    }

    void CalculateResidualVector(View_N residual_vector) {
        // Calculate nodal force vectors
        Kokkos::parallel_for(
            "CalculateNodeForces", this->num_beams_,
            CalculateNodeForces{
                this->elem_indices, this->qp_weight, this->qp_jacobian, this->shape_interp,
                this->shape_deriv, this->qp_Fc, this->qp_Fd, this->qp_Fi, this->qp_Fg, this->node_FE,
                this->node_FI, this->node_FG}
        );

        // Sum node force vectors into residual vector
        // TODO: this should be a reduce operation, but can't figure out how to do it
        for (size_t i = 0; i < this->num_nodes_; ++i) {
            auto i_rv_start = 6 * this->node_state_indices(i);
            for (size_t j = 0; j < 6; j++) {
                residual_vector(i_rv_start + j) += this->node_FE(i, j) + this->node_FI(i, j) -
                                                   this->node_FX(i, j) - this->node_FG(i, j);
            }
        }

        // Kokkos::parallel_reduce(
        //     this->num_nodes_,
        //     AssembleResidualVector(
        //         residual_vector.extent(0), this->node_state_indices, this->node_force_elastic,
        //         this->node_force_inertial, this->node_force_gravity, this->node_force_external
        //     ),
        //     residual_vector
        // );
    }
};

}  // namespace oturb