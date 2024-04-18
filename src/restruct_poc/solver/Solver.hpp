#pragma once

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ConstraintInput.hpp"
#include "Constraints.hpp"
#include "State.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Solver {
    bool is_dynamic_solve;
    int max_iter;
    double h;
    double alpha_m;
    double alpha_f;
    double gamma;
    double beta;
    double gamma_prime;
    double beta_prime;
    double conditioner;
    int num_system_nodes;
    int num_system_dofs;
    int num_constraint_nodes;
    int num_constraint_dofs;
    int num_dofs;
    View_NxN M;   // Mass matrix
    View_NxN G;   // Gyroscopic matrix
    View_NxN K;   // Stiffness matrix
    View_NxN T;   // Tangent matrix
    View_NxN St;  // Iteration matrix
    Kokkos::View<double**, Kokkos::LayoutLeft> St_left;
    Kokkos::View<int*, Kokkos::LayoutLeft> IPIV;
    View_N R;  // System residual vector
    View_N x;  // System solution vector
    State state;
    Constraints constraints;
    std::vector<double> convergence_err;

    Solver() {}
    Solver(
        bool is_dynamic_solve_, int max_iter_, double h_, double rho_inf, int num_system_nodes_,
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
          St_left("St_left", num_dofs, num_dofs),
          IPIV("IPIV", num_dofs),
          R("R", num_dofs),
          x("x", num_dofs),
          state(num_system_nodes, num_constraint_nodes, q_, v_, vd_),
          constraints(constraint_inputs, num_system_nodes),
          convergence_err(max_iter) {}
};

}  // namespace openturbine
