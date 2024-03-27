#pragma once

#include "types.hpp"

namespace oturb {

struct Constraints {
    View_N residual_vector;
    View_NxN gradient_matrix;
    Constraints() {}
    Constraints(size_t num_system_dofs, size_t num_constraint_dofs)
        : residual_vector("residual_vector", num_constraint_dofs),
          gradient_matrix("gradient_matrix", num_constraint_dofs, num_system_dofs) {}
};

struct State {
    View_Nx6 q_delta;
    View_Nx7 q_prev;
    View_Nx7 q;
    View_Nx6 v;
    View_Nx6 vd;
    View_Nx6 a;
    View_Nx6 lambda;
    State() {}
    State(size_t num_nodes)
        : q_delta("q_delta", num_nodes),
          q_prev("q_prev", num_nodes),
          q("q", num_nodes),
          v("v", num_nodes),
          vd("vd", num_nodes),
          a("a", num_nodes),
          lambda("lambda", num_nodes) {}
};

struct Solver {
    size_t num_system_dofs;
    size_t num_constraint_dofs;
    size_t num_dofs;
    View_NxN iteration_matrix;
    View_N residual_vector;
    State state;
    Constraints constraints;

    Solver() {}
    Solver(size_t num_system_nodes, size_t num_constraint_nodes)
        : num_system_dofs(num_system_nodes * kLieAlgebraComponents),
          num_constraint_dofs(num_constraint_nodes * kLieAlgebraComponents),
          num_dofs(num_system_dofs + num_constraint_dofs),
          iteration_matrix("iteration_matrix", num_dofs, num_dofs),
          residual_vector("residual_vector", num_dofs),
          state(num_system_nodes),
          constraints(num_system_dofs, num_constraint_dofs) {}
};

}  // namespace oturb
