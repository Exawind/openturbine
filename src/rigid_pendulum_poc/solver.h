#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Dense>

namespace openturbine::rigid_pendulum {

enum class SolverType {
    kDirectLinearSolver,
    kIterativeLinearSolver,
    kIterativeNonlinearSolver
};

// TODO Think about the architecture some more, should all solvers inherit this interface?
/// @brief An abstract class to define the interface for linear system solvers
class LinearSystemSolver {
public:
    // Ctor is not needed but a dtor is necessary for a pure virtual class
    virtual ~LinearSystemSolver() = default;

    virtual Eigen::VectorXd Solve(const Eigen::MatrixXd&, const Eigen::VectorXd&) const = 0;
};

/// @brief LDLT solver for a linear systems of equations
/// @details This class uses Eigen's LDLT solver to solve a linear system of
/// equations of the form Ax = b, where A is a symmetric positive definite
/// matrix, x is the vector of unknowns, and b is the load vector.
class LDLTSolver : public LinearSystemSolver {
public:
    LDLTSolver() {}

    inline Eigen::VectorXd Solve(const Eigen::MatrixXd& stiffness_matrix,
                                 const Eigen::VectorXd& load_vector) const {
        Eigen::VectorXd displacements = stiffness_matrix.ldlt().solve(load_vector);
        return displacements;
    }
};

// TODO Introduce an abstract non-linear solver class to facilitate interface inheritance
/// @brief Generalized-alpha solver for constrained, nonlinear structural dynamics
class GeneralizedAlphaSolver {
public:
    GeneralizedAlphaSolver(std::unique_ptr<LinearSystemSolver>);

    void Solve();

private:
    std::unique_ptr<LinearSystemSolver> linear_system_solver_;

    // Implement generalized-alpha scheme for the rigid pendulum problem
    void AlphaStep(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&,
                   const Eigen::VectorXd&);
};

}  // namespace openturbine::rigid_pendulum
