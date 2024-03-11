#include "src/gebt_poc/force.h"

#include <iostream>

#include <KokkosBlas.hpp>

namespace openturbine::gebt_poc {

struct InitializeGeneralizedForces {
    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        generalized_forces_(0) = forces.GetXComponent();
        generalized_forces_(1) = forces.GetYComponent();
        generalized_forces_(2) = forces.GetZComponent();
        generalized_forces_(3) = moments.GetXComponent();
        generalized_forces_(4) = moments.GetYComponent();
        generalized_forces_(5) = moments.GetZComponent();
    }
    gen_alpha_solver::Vector forces;
    gen_alpha_solver::Vector moments;
    View1D_LieAlgebra generalized_forces_;
};

GeneralizedForces::GeneralizedForces(
    const gen_alpha_solver::Vector& forces, const gen_alpha_solver::Vector& moments, size_t node
)
    : forces_(forces), moments_(moments), generalized_forces_("generalized_forces"), node_(node) {
    Kokkos::parallel_for(1, InitializeGeneralizedForces{forces, moments, generalized_forces_});
}

GeneralizedForces::GeneralizedForces(View1D_LieAlgebra generalized_forces, size_t node)
    : generalized_forces_("generalized_forces"), node_(node) {
    Kokkos::deep_copy(generalized_forces_, generalized_forces);

    auto forces = Kokkos::subview(generalized_forces_, Kokkos::make_pair(0, 3));
    auto forces_host = Kokkos::create_mirror(forces);
    Kokkos::deep_copy(forces_host, forces);
    this->forces_ = gen_alpha_solver::Vector(forces_host(0), forces_host(1), forces_host(2));

    auto moments = Kokkos::subview(generalized_forces_, Kokkos::make_pair(3, 6));
    auto moments_host = Kokkos::create_mirror(moments);
    Kokkos::deep_copy(moments_host, moments);
    this->moments_ = gen_alpha_solver::Vector(moments_host(0), moments_host(1), moments_host(2));
}

TimeVaryingForces::TimeVaryingForces(
    std::function<View1D_LieAlgebra(double time)> generalized_forces_function, size_t node
)
    : function_(generalized_forces_function), node_(node) {
}

View1D_LieAlgebra TimeVaryingForces::GetGeneralizedForces(double time) const {
    return function_(time);
}

}  // namespace openturbine::gebt_poc
