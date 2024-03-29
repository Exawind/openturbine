#pragma once

#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/vector.h"

namespace openturbine::gebt_poc {

/// Create an abstract class for providing common interface for different types of forces
class Forces {
public:
    virtual ~Forces() = default;

    /// Returns the 6 x 1 force vector
    virtual View1D_LieAlgebra GetGeneralizedForces(double /*time*/) const = 0;

    /// Returns the node number on which the force is being applied
    virtual size_t GetNode() const = 0;
};

/// Class for managing the generalized forces applied on a dynamic system
class GeneralizedForces : public Forces {
public:
    /// Default constructor that initializes all generalized forces to zero
    GeneralizedForces(
        const gen_alpha_solver::Vector& forces = gen_alpha_solver::Vector(),
        const gen_alpha_solver::Vector& moments = gen_alpha_solver::Vector(), size_t node = 1
    );

    /// Constructor that initializes the generalized forces to the given vectors
    GeneralizedForces(View1D_LieAlgebra generalized_forces, size_t node);

    /// Returns the 3 x 1 force vector
    inline gen_alpha_solver::Vector GetForces() const { return forces_; }

    /// Returns the 3 x 1 moment vector
    inline gen_alpha_solver::Vector GetMoments() const { return moments_; }

    /// Returns the 6 x 1 generalized forces vector
    inline View1D_LieAlgebra GetGeneralizedForces([[maybe_unused]] double time = 0.) const override {
        return generalized_forces_;
    }

    /// Returns the node number on which the generalized forces are applied
    inline size_t GetNode() const override { return node_; }

private:
    gen_alpha_solver::Vector forces_;   //< force vector
    gen_alpha_solver::Vector moments_;  //< moment vector
    View1D_LieAlgebra
        generalized_forces_;  //< Generalized forces (combined forces and moments vector)
    size_t node_;             //< Node number on which the generalized forces are applied
};

/// Class for managing the time-varying generalized forces applied on a dynamic system
class TimeVaryingForces : public Forces {
public:
    /// Constructor that creates the generalized forces from a given function of time
    TimeVaryingForces(
        std::function<View1D_LieAlgebra(double time)> generalized_forces_function, size_t node
    );

    /// Returns the 6 x 1 generalized forces vector
    View1D_LieAlgebra GetGeneralizedForces(double time = 0.) const override;

    /// Returns the node number on which the generalized forces are applied
    inline size_t GetNode() const override { return node_; }

private:
    std::function<View1D_LieAlgebra(double)> function_;
    size_t node_;
};

}  // namespace openturbine::gebt_poc
