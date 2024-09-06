#pragma once

#include <Kokkos_Core.hpp>

#include "src/types.hpp"

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
    View_N R;
    double conditioner;

    KOKKOS_FUNCTION
    void operator()(int i) const { R(i) *= conditioner; }
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
    View_N x;

    KOKKOS_FUNCTION
    void operator()(const int i) const {
        if (static_cast<size_t>(i) >= num_system_dofs) {
            x(i) /= conditioner;
        }
    }
};

}  // namespace openturbine
