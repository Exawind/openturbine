#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

struct PreconditionSt {
    View_NxN St;
    double conditioner;

    KOKKOS_FUNCTION
    void operator()(int i, int j) const { St(i, j) *= conditioner; }
};

struct PostconditionSt {
    View_NxN St;
    double conditioner;

    KOKKOS_FUNCTION
    void operator()(int i, int j) const { St(i, j) /= conditioner; }
};

struct ConditionR {
    double conditioner;
    Kokkos::View<double* [1], Kokkos::LayoutLeft> R;

    KOKKOS_FUNCTION
    void operator()(int i) const { R(i, 0) *= conditioner; }
};

struct ConditionSystem {
    int num_system_dofs;
    int num_dofs;
    double conditioner;
    View_NxN St;
    View_N R;

    KOKKOS_FUNCTION
    void operator()(const int) const {
        for (int i = 0; i < num_system_dofs; ++i) {
            for (int j = 0; j < num_dofs; ++j) {
                St(i, j) *= conditioner;
            }
        }

        for (int i = 0; i < num_dofs; ++i) {
            for (int j = num_system_dofs; j < num_dofs; ++j) {
                St(i, j) /= conditioner;
            }
        }

        for (int i = 0; i < num_system_dofs; ++i) {
            R(i) *= conditioner;
        }
    }
};

struct UnconditionSolution {
    size_t num_system_dofs;
    double conditioner;
    Kokkos::View<double* [1], Kokkos::LayoutLeft> x;

    KOKKOS_FUNCTION
    void operator()(size_t i) const { x(i + num_system_dofs, 0) /= conditioner; }
};

}  // namespace openturbine
