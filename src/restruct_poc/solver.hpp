#pragma once

#include "types.hpp"

namespace oturb {

struct Constraints {
    size_t num_constraint_nodes;
    Kokkos::View<size_t* [2]> node_indices;
    View_N Phi;
    View_NxN B;
    View_Nx3 X0;
    View_Nx7 u;
    Constraints() {}
    Constraints(size_t num_system_nodes, size_t num_constraint_nodes_)
        : num_constraint_nodes(num_constraint_nodes_),
          node_indices("node_indices", num_constraint_nodes),
          Phi("residual_vector", num_constraint_nodes * kLieAlgebraComponents),
          B("gradient_matrix", num_constraint_nodes * kLieAlgebraComponents,
            num_system_nodes * kLieAlgebraComponents),
          X0("X0", num_constraint_nodes),
          u("u", num_constraint_nodes) {}
};

struct State {
    View_Nx6 q_delta;
    View_Nx7 q_prev;
    View_Nx7 q;
    View_Nx6 v;
    View_Nx6 vd;
    View_Nx6 a;
    View_N lambda;
    State() {}
    State(size_t num_system_nodes, size_t num_constraint_nodes)
        : q_delta("q_delta", num_system_nodes),
          q_prev("q_prev", num_system_nodes),
          q("q", num_system_nodes),
          v("v", num_system_nodes),
          vd("vd", num_system_nodes),
          a("a", num_system_nodes),
          lambda("lambda", num_constraint_nodes * kLieAlgebraComponents) {}
    State(
        size_t num_system_nodes, size_t num_constraint_nodes, std::vector<std::array<double, 7>> q_,
        std::vector<std::array<double, 7>> v_, std::vector<std::array<double, 7>> vd_
    )
        : State(num_system_nodes, num_constraint_nodes) {
        // Create mirror of state views
        auto host_q = Kokkos::create_mirror(this->q);
        auto host_v = Kokkos::create_mirror(this->v);
        auto host_vd = Kokkos::create_mirror(this->vd);

        // Loop through number of nodes and copy data to host view
        for (size_t i = 0; i < num_system_nodes; ++i) {
            for (size_t j = 0; j < kLieGroupComponents; ++j) {
                host_q(i, j) = q_[i][j];
            }
            for (size_t j = 0; j < kLieAlgebraComponents; ++j) {
                host_v(i, j) = v_[i][j];
                host_vd(i, j) = vd_[i][j];
            }
        }

        // Transfer host view data to state views
        Kokkos::deep_copy(this->q, host_q);
        Kokkos::deep_copy(this->v, host_v);
        Kokkos::deep_copy(this->vd, host_vd);

        // Copy q to q_previous
        Kokkos::deep_copy(this->q_prev, this->q);
    }
};

struct Solver {
    bool is_dynamic_solve;
    size_t max_iter;
    double h;
    double alpha_m;
    double alpha_f;
    double gamma;
    double beta;
    double gamma_prime;
    double beta_prime;
    double conditioner;
    size_t num_system_nodes;
    size_t num_system_dofs;
    size_t num_constraint_nodes;
    size_t num_constraint_dofs;
    size_t num_dofs;
    View_NxN M;   // Mass matrix
    View_NxN G;   // Gyroscopic matrix
    View_NxN K;   // Stiffness matrix
    View_NxN T;   // Tangent matrix
    View_NxN St;  // Iteration matrix
    View_N R;     // System residual vector
    View_N x;     // System solution vector
    State state;
    Constraints constraints;

    Solver() {}
    Solver(
        bool is_dynamic_solve_, size_t max_iter_, double h_, double rho_inf,
        size_t num_system_nodes_, size_t num_constraint_nodes_
    )
        : is_dynamic_solve(is_dynamic_solve_),
          max_iter(max_iter_),
          h(h_),
          alpha_m((2. * rho_inf - 1.) / (rho_inf + 1.)),
          alpha_f(rho_inf / (rho_inf + 1.)),
          gamma(0.5 + alpha_f - alpha_m),
          beta(0.25 * (gamma + 0.5) * (gamma + 0.5)),
          gamma_prime(gamma / (h * beta)),
          beta_prime((1. - alpha_m) / (h * h * beta * (1. - alpha_f))),
          conditioner(beta * h * h),
          num_system_nodes(num_system_nodes_),
          num_system_dofs(num_system_nodes_ * kLieAlgebraComponents),
          num_constraint_nodes(num_constraint_nodes_),
          num_constraint_dofs(num_constraint_nodes_ * kLieAlgebraComponents),
          num_dofs(num_system_dofs + num_constraint_dofs),
          M("M", num_system_dofs, num_system_dofs),
          G("G", num_system_dofs, num_system_dofs),
          K("K", num_system_dofs, num_system_dofs),
          T("T", num_system_dofs, num_system_dofs),
          St("St", num_dofs, num_dofs),
          R("R", num_system_dofs),
          x("x", num_dofs),
          state(num_system_nodes_, num_constraint_nodes_),
          constraints(num_system_nodes, num_constraint_nodes) {}
};

}  // namespace oturb
