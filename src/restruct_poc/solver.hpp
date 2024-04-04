#pragma once

#include "beams.hpp"
#include "state.hpp"
#include "types.hpp"

namespace openturbine {

struct ConstraintInput {
    size_t base_node_index;
    size_t constrained_node_index;
    ConstraintInput(size_t node1, size_t node2)
        : base_node_index(node1), constrained_node_index(node2) {}
};

struct Constraints {
    struct NodeIndices {
        size_t base_node_index;
        size_t constrained_node_index;
    };

    size_t num_constraint_nodes;
    Kokkos::View<NodeIndices*> node_indices;
    View_N Phi;
    View_NxN B;
    View_Nx3 X0;
    View_Nx7 u;
    Constraints() {}
    Constraints(std::vector<ConstraintInput> inputs, size_t num_system_nodes)
        : num_constraint_nodes(inputs.size()),
          node_indices("node_indices", num_constraint_nodes),
          Phi("residual_vector", num_constraint_nodes * kLieAlgebraComponents),
          B("gradient_matrix", num_constraint_nodes * kLieAlgebraComponents,
            num_system_nodes * kLieAlgebraComponents),
          X0("X0", num_constraint_nodes),
          u("u", num_constraint_nodes) {
        // Copy constraint input data to views
        auto host_node_indices = Kokkos::create_mirror(this->node_indices);
        for (size_t i = 0; i < inputs.size(); ++i) {
            host_node_indices(i).base_node_index = inputs[i].base_node_index;
            host_node_indices(i).constrained_node_index = inputs[i].constrained_node_index;
        }
        Kokkos::deep_copy(this->node_indices, host_node_indices);
        // Initialize rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    void UpdateDisplacement(size_t index, std::array<double, kLieGroupComponents> u_) {
        auto host_u = Kokkos::create_mirror(this->u);
        for (size_t i = 0; i < kLieGroupComponents; ++i) {
            host_u(index, i) = u_[i];
        }
        Kokkos::deep_copy(this->u, host_u);
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
    double convergence_err;
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
        size_t num_system_nodes_,
        std::vector<ConstraintInput> constraint_inputs = std::vector<ConstraintInput>(),
        std::vector<std::array<double, 7>> q_ = std::vector<std::array<double, 7>>(),
        std::vector<std::array<double, 6>> v_ = std::vector<std::array<double, 6>>(),
        std::vector<std::array<double, 6>> vd_ = std::vector<std::array<double, 6>>()
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
          convergence_err(0.),
          num_system_nodes(num_system_nodes_),
          num_system_dofs(num_system_nodes * kLieAlgebraComponents),
          num_constraint_nodes(constraint_inputs.size()),
          num_constraint_dofs(num_constraint_nodes * kLieAlgebraComponents),
          num_dofs(num_system_dofs + num_constraint_dofs),
          M("M", num_system_dofs, num_system_dofs),
          G("G", num_system_dofs, num_system_dofs),
          K("K", num_system_dofs, num_system_dofs),
          T("T", num_system_dofs, num_system_dofs),
          St("St", num_dofs, num_dofs),
          R("R", num_dofs),
          x("x", num_dofs),
          state(num_system_nodes, num_constraint_nodes, q_, v_, vd_),
          constraints(constraint_inputs, num_system_nodes) {}
};

void PredictNextState(Solver& solver);
void InitializeConstraints(Solver& solver, Beams& beams);
void UpdateStatePrediction(Solver& solver, View_N x_system, View_N x_lambda);
template <typename Subview_NxN, typename Subview_N>
void AssembleSystem(Solver& solver, Beams& beams, Subview_NxN St, Subview_N R);
template <typename Subview_NxN, typename Subview_N>
void AssembleConstraints(
    Solver& solver, Subview_NxN St_12, Subview_NxN St_21, Subview_N R_system, Subview_N R_lambda
);
bool Step(Solver& solver, Beams& beams);
void SolveSystem(Solver& solver);
double CalculateConvergenceError(Solver& solver);

}  // namespace openturbine
