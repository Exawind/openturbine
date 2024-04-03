#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>

#include "src/gebt_poc/force.h"
#include "src/gebt_poc/linearization_parameters.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/state.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

/// Calculates the constraint gradient matrix for the clamped beam problem
void BMatrix(View2D constraints_gradient_matrix);

/*!
 * Calculates the residual vector and iteration matrix for a static beam element
 */
class StaticBeamLinearizationParameters : public LinearizationParameters {
public:
    /// Define a static beam element with the given position vector for the nodes, 6x6
    /// stiffness matrix, and a quadrature rule
    StaticBeamLinearizationParameters(
        LieGroupFieldView position_vectors, View2D_6x6 stiffness_matrix,
        UserDefinedQuadrature quadrature
    );

    virtual void ResidualVector(
        LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type velocity,
        LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_multipliers,
        const gen_alpha_solver::TimeStepper& time_stepper, View1D residual_vector
    ) override;

    virtual void IterationMatrix(
        double h, double beta_prime, double gamma_prime, LieGroupFieldView::const_type gen_coords,
        LieAlgebraFieldView::const_type delta_gen_coords, LieAlgebraFieldView::const_type velocity,
        LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_multipliers,
        View2D iteration_matrix
    ) override;

    /// Tangent operator for a single node of the static beam element
    void TangentOperator(
        LieAlgebraFieldView::const_type delta_gen_coords, double h, View2D tangent_operator
    ) {
        using member_type = Kokkos::TeamPolicy<>::member_type;
        using no_transpose = KokkosBatched::Trans::NoTranspose;
        using unblocked = KokkosBatched::Algo::Gemm::Unblocked;
        using gemm =
            KokkosBatched::TeamVectorGemm<member_type, no_transpose, no_transpose, unblocked>;
        using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
        using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
        using ScratchView1D_Vector = Kokkos::View<double[3], scratch_space, unmanaged_memory>;
        using ScratchView2D_6x6 = Kokkos::View<double[6][6], scratch_space, unmanaged_memory>;
        using ScratchView2D_3x3 = Kokkos::View<double[3][3], scratch_space, unmanaged_memory>;

        const auto n_nodes = delta_gen_coords.extent(0);
        auto policy = Kokkos::TeamPolicy<>(n_nodes, Kokkos::AUTO(), Kokkos::AUTO());
        const auto scratch_size = ScratchView1D_Vector::shmem_size() +
                                  ScratchView2D_6x6::shmem_size() +
                                  2 * ScratchView2D_3x3::shmem_size();
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

        Kokkos::deep_copy(tangent_operator, 0.0);
        Kokkos::parallel_for(
            policy,
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
                const auto node = team_member.league_rank();
                const auto delta_gen_coords_node = [&](auto& member
                                                   ) -> ScratchView1D_Vector::const_type {
                    auto nodal_delta = ScratchView1D_Vector(member.team_scratch(0));
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 3), [&](std::size_t i) {
                        nodal_delta(i) = delta_gen_coords(node, 3 + i) / h;
                    });
                    return nodal_delta;
                }(team_member);
                const auto tangent_operator_node = [&](auto& member) {
                    auto initial_operator = ScratchView2D_6x6(member.team_scratch(0));
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorMDRange(member, 6, 6),
                        [&](std::size_t i, std::size_t j) {
                            initial_operator(i, j) = i == j;
                        }
                    );
                    return initial_operator;
                }(team_member);
                team_member.team_barrier();
                const double phi = std::sqrt(
                    delta_gen_coords_node(0) * delta_gen_coords_node(0) +
                    delta_gen_coords_node(1) * delta_gen_coords_node(1) +
                    delta_gen_coords_node(2) * delta_gen_coords_node(2)
                );
                if (phi > Tolerance) {
                    const auto psi_cross_prod_matrix = [&](auto& member,
                                                           ScratchView1D_Vector::const_type vector) {
                        auto output = ScratchView2D_3x3(member.team_scratch(0));
                        Kokkos::single(Kokkos::PerTeam(member), [&]() {
                            output(0, 0) = 0.;
                            output(0, 1) = -vector(2);
                            output(0, 2) = vector(1);
                            output(1, 0) = vector(2);
                            output(1, 1) = 0.;
                            output(1, 2) = -vector(0);
                            output(2, 0) = -vector(1);
                            output(2, 1) = vector(0);
                            output(2, 2) = 0.;
                        });
                        return output;
                    }(team_member, delta_gen_coords_node);
                    team_member.team_barrier();
                    const auto psi_times_psi = [&](auto& member, ScratchView2D_3x3::const_type psi) {
                        auto psi_x_psi = ScratchView2D_3x3(member.team_scratch(0));
                        gemm::invoke(member, 1., psi, psi, 0., psi_x_psi);
                        return psi_x_psi;
                    }(team_member, psi_cross_prod_matrix);
                    const auto factor_1 = (std::cos(phi) - 1.0) / (phi * phi);
                    const auto factor_2 = (1.0 - std::sin(phi) / phi) / (phi * phi);
                    team_member.team_barrier();
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorMDRange(team_member, 3, 3),
                        [&](std::size_t i, std::size_t j) {
                            tangent_operator_node(3 + i, 3 + j) +=
                                factor_1 * psi_cross_prod_matrix(i, j) +
                                factor_2 * psi_times_psi(i, j);
                        }
                    );
                }

                team_member.team_barrier();
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorMDRange(team_member, 6, 6),
                    [&](std::size_t i, std::size_t j) {
                        tangent_operator(node * 6 + i, node * 6 + j) = tangent_operator_node(i, j);
                    }
                );
            }
        );
    }

private:
    LieGroupFieldView position_vectors_;
    View2D_6x6 stiffness_matrix_;
    UserDefinedQuadrature quadrature_;
};

}  // namespace openturbine::gebt_poc
