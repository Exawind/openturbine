#include "src/gebt_poc/force.h"

#include <iostream>

#include <KokkosBlas.hpp>

namespace openturbine::gebt_poc {

GeneralizedForces::GeneralizedForces(
    const gen_alpha_solver::Vector& forces, const gen_alpha_solver::Vector& moments, size_t node
)
    : forces_(forces), moments_(moments), generalized_forces_("generalized_forces"), node_(node) {
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            generalized_forces_(0) = forces.GetXComponent();
            generalized_forces_(1) = forces.GetYComponent();
            generalized_forces_(2) = forces.GetZComponent();
            generalized_forces_(3) = moments.GetXComponent();
            generalized_forces_(4) = moments.GetYComponent();
            generalized_forces_(5) = moments.GetZComponent();
        }
    );
}

GeneralizedForces::GeneralizedForces(Kokkos::View<double[6]> generalized_forces, size_t node)
    : generalized_forces_("generalized_forces"), node_(node) {
    Kokkos::deep_copy(generalized_forces_, generalized_forces);

    auto forces = Kokkos::subview(generalized_forces_, Kokkos::make_pair(0, 3));
    auto moments = Kokkos::subview(generalized_forces_, Kokkos::make_pair(3, 6));
    this->forces_ = gen_alpha_solver::Vector(forces(0), forces(1), forces(2));
    this->moments_ = gen_alpha_solver::Vector(moments(0), moments(1), moments(2));
}

TimeVaryingForces::TimeVaryingForces(
    std::function<Kokkos::View<double[6]>(double time)> generalized_forces_function, size_t node
)
    : function_(generalized_forces_function), node_(node) {
}

Kokkos::View<double*> TimeVaryingForces::GetGeneralizedForces(double time) const {
    return function_(time);
}

}  // namespace openturbine::gebt_poc
