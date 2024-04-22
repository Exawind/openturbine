#pragma once

#include <Kokkos_Core.hpp>

#include "Constraints.hpp"

#include "src/restruct_poc/types.hpp"
#include "src/restruct_poc/QuaternionOperations.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<Constraints::NodeIndices*>::const_type node_indices;
    View_Nx3::const_type constraint_X0;
    View_Nx7::const_type constraint_u;
    View_Nx7::const_type node_u;
    View_N Phi;
    View_NxN B;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto i_node1 = node_indices(i_constraint).base_node_index;
        auto i_node2 = node_indices(i_constraint).constrained_node_index;

        auto r1_data = [=](){
            if (i_node1 == -1) {
                return Kokkos::Array<double, 4>{constraint_u(i_constraint, 3), constraint_u(i_constraint, 4), constraint_u(i_constraint, 5), constraint_u(i_constraint, 6)};
            }
            return Kokkos::Array<double, 4>{node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)};
        }();
        auto R1 = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{r1_data.data()};

        auto r2_data = Kokkos::Array<double, 4>{node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)};
        auto R2 = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{r2_data.data()};

        auto x0_data = Kokkos::Array<double, 3>{constraint_X0(i_constraint, 0), constraint_X0(i_constraint, 1), constraint_X0(i_constraint, 2)};
        auto X0 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{x0_data.data()};

        auto u2_data = Kokkos::Array<double, 3>{node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2)};
        auto u2 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{u2_data.data()};

        auto phi_x_data = Kokkos::Array<double, 3>{};
        auto Phi_x = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{phi_x_data.data()};
        QuaternionRotateVector(R1, X0, Phi_x);
        KokkosBlas::serial_axpy(-1., X0, Phi_x);
        KokkosBlas::serial_axpy(-1., u2, Phi_x);

        auto i_row = i_constraint * kLieAlgebraComponents;
        auto i_col = i_node2 * kLieAlgebraComponents;

        auto inverse_data = Kokkos::Array<double, 4>{};
        auto inverse = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{inverse_data.data()};
        QuaternionInverse(R1, inverse);

        auto R2_R1_Inverse_data = Kokkos::Array<double, 4>{};
        auto R2_R1_Inverse = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2_R1_Inverse_data.data()};
        QuaternionCompose(R2, inverse, R2_R1_Inverse);
      
        auto rotation_data = Kokkos::Array<double, 9>{};
        auto rotation = Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{rotation_data.data()};
        QuaternionToRotationMatrix(R2_R1_Inverse, rotation);

        auto Phi_p_data = Kokkos::Array<double, 3>{};
        auto Phi_p = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{Phi_p_data.data()};
        ComputeAxialVector(rotation, Phi_p);

        Phi(i_row + 0) = -Phi_x(0);
        Phi(i_row + 1) = -Phi_x(1);
        Phi(i_row + 2) = -Phi_x(2);

        Phi(i_row + 3) = Phi_p(0);
        Phi(i_row + 4) = Phi_p(1);
        Phi(i_row + 5) = Phi_p(2);

        QuaternionInverse(R2, inverse);
        QuaternionCompose(R1, inverse, R2_R1_Inverse);
        QuaternionToRotationMatrix(R2_R1_Inverse, rotation);
        auto trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);

        for (int i = 0; i < 3; ++i) {
            B(i_row + i, i_col + i) = 1.;
        }
        for (int i = 0; i < 3; ++i) {
            B(i_row + i + 3, i_col + i + 3) = trace / 2.;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B(i_row + i + 3, i_col + j + 3) -= rotation(i, j) / 2.;
            }
        }
    }
};

}  // namespace openturbine
