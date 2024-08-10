#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateInertiaMatrixElement {
    int i_elem;
    size_t num_nodes;
    size_t num_qps;
    size_t first_node;
    size_t first_qp;
    View_N::const_type qp_weight_;
    View_N::const_type qp_jacobian_;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_interp_;
    Kokkos::View<double* [6][6]>::const_type qp_Muu_;
    Kokkos::View<double* [6][6]>::const_type qp_Guu_;
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index) const {
        using simd_type = Kokkos::Experimental::native_simd<double>;
        using mask_type = Kokkos::Experimental::native_simd_mask<double>;
        using tag_type = Kokkos::Experimental::element_aligned_tag;
        constexpr auto width = simd_type::size();
        for (auto j_index = 0U; j_index < num_nodes; j_index += width) {
            auto mask = mask_type([j_index, num_nodes = this->num_nodes](size_t lane) {
                return j_index + lane < num_nodes;
            });
            auto local_M_data = Kokkos::Array<simd_type, 36>{};
            const auto local_M = Kokkos::View<simd_type[6][6]>(local_M_data.data());
            for (auto k = 0U; k < num_qps; ++k) {
                const auto w = qp_weight_(k);
                const auto jacobian = qp_jacobian_(k);
                const auto phi_i = shape_interp_(i_index, k);
                auto phi_j = simd_type{};
                Kokkos::Experimental::where(mask, phi_j)
                    .copy_from(&shape_interp_(j_index, k), tag_type());
                const auto coeff = w * phi_i * phi_j * jacobian;
                for (auto m = 0U; m < 6U; ++m) {
                    for (auto n = 0U; n < 6U; ++n) {
                        local_M(m, n) += coeff * (beta_prime_ * qp_Muu_(k, m, n) +
                                                  gamma_prime_ * qp_Guu_(k, m, n));
                    }
                }
            }
            for (auto m = 0U; m < 6U; ++m) {
                for (auto lane = 0U; lane < width && mask[lane]; ++lane) {
                    for (auto n = 0U; n < 6U; ++n) {
                        gbl_M_(i_elem, i_index * 6 + m, (j_index + lane) * 6 + n) =
                            local_M(m, n)[lane];
                    }
                }
            }
        }
    }
};

struct IntegrateInertiaMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type qp_weight_;
    View_NxN::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double** [6][6]>::const_type qp_Muu_;
    Kokkos::View<double** [6][6]>::const_type qp_Guu_;
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type& member) const {
        const auto i_elem = member.league_rank();
        const auto idx = elem_indices(i_elem);
        const auto shape_interp = Kokkos::View<double**, Kokkos::LayoutLeft>(
            member.team_scratch(1), idx.num_nodes, idx.num_qps
        );

        const auto qp_weight = Kokkos::View<double*>(member.team_scratch(1), idx.num_qps);
        const auto qp_jacobian = Kokkos::View<double*>(member.team_scratch(1), idx.num_qps);

        const auto qp_Muu = Kokkos::View<double* [6][6]>(member.team_scratch(1), idx.num_qps);
        const auto qp_Guu = Kokkos::View<double* [6][6]>(member.team_scratch(1), idx.num_qps);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, idx.num_qps), [=](size_t k) {
            for (auto i = 0U; i < idx.num_nodes; ++i) {
                shape_interp(i, k) = shape_interp_(i_elem, i, k);
            }
            qp_weight(k) = qp_weight_(i_elem, k);
            qp_jacobian(k) = qp_jacobian_(i_elem, k);
            for (auto m = 0U; m < 6U; ++m) {
                for (auto n = 0U; n < 6U; ++n) {
                    qp_Muu(k, m, n) = qp_Muu_(i_elem, k, m, n);
                    qp_Guu(k, m, n) = qp_Guu_(i_elem, k, m, n);
                }
            }
        });
        member.team_barrier();

        const auto node_range = Kokkos::TeamThreadRange(member, idx.num_nodes);
        const auto element_integrator = IntegrateInertiaMatrixElement{
            i_elem,
            idx.num_nodes,
            idx.num_qps,
            idx.node_range.first,
            idx.qp_range.first,
            qp_weight,
            qp_jacobian,
            shape_interp,
            qp_Muu,
            qp_Guu,
            beta_prime_,
            gamma_prime_,
            gbl_M_};
        Kokkos::parallel_for(node_range, element_integrator);
    }
};
}  // namespace openturbine
