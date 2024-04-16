#pragma once

#include <Kokkos_Core.hpp>
#include "types.hpp"
#include "beams_data.hpp"

namespace openturbine {

struct InterpolateQPU {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_interp_;
    View_Nx7::const_type node_u_;
    View_Nx3 qp_u_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }
        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPU_Prime {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_deriv_;
    View_N::const_type qp_jacobian_;
    View_Nx7::const_type node_u_;
    View_Nx3 qp_u_prime_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto jacobian = qp_jacobian_(j);
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto dphi = shape_deriv_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_(i, k) * dphi / jacobian;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_prime_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        const auto jacobian = qp_jacobian_(j);
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto dphi = shape_deriv_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_(i, k) * dphi / jacobian;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u_prime_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPR {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_interp_;
    View_Nx7::const_type node_u_;
    View_Nx4 qp_r_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 4>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 4; ++k) {
                    local_total[k] += node_u_(i, k + 3) * phi;
                }
            }
            const auto length = Kokkos::sqrt(
                local_total[0] * local_total[0] + local_total[1] * local_total[1] +
                local_total[2] * local_total[2] + local_total[3] * local_total[3]
            );
            static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
            if (length == 0.) {
                local_total = length_zero_result;
            }
            for (int k = 0; k < 4; ++k) {
                qp_r_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 4>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 4; ++k) {
                local_total[k] += node_u_(i, k + 3) * phi;
            }
        }
        const auto length = Kokkos::sqrt(
            local_total[0] * local_total[0] + local_total[1] * local_total[1] +
            local_total[2] * local_total[2] + local_total[3] * local_total[3]
        );
        static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
        if (length == 0.) {
            local_total = length_zero_result;
        }
        for (int k = 0; k < 4; ++k) {
            qp_r_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPR_Prime {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_deriv_;
    View_N::const_type qp_jacobian_;
    View_Nx7::const_type node_u_;
    View_Nx4 qp_r_prime_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto jacobian = qp_jacobian_(j);
            auto local_total = Kokkos::Array<double, 4>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto dphi = shape_deriv_(i, j_index);
                for (int k = 0; k < 4; ++k) {
                    local_total[k] += node_u_(i, k + 3) * dphi / jacobian;
                }
            }
            for (int k = 0; k < 4; ++k) {
                qp_r_prime_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        const auto jacobian = qp_jacobian_(j);
        auto local_total = Kokkos::Array<double, 4>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto dphi = shape_deriv_(i, j_index);
            for (int k = 0; k < 4; ++k) {
                local_total[k] += node_u_(i, k + 3) * dphi / jacobian;
            }
        }
        for (int k = 0; k < 4; ++k) {
            qp_r_prime_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPState {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_interp_;
    View_NxN::const_type shape_deriv_;
    View_N::const_type qp_jacobian_;
    View_Nx7::const_type node_u_;
    View_Nx3 qp_u_;
    View_Nx3 qp_u_prime_;
    View_Nx4 qp_r_;
    View_Nx4 qp_r_prime_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_(j, k) = local_total[k];
            }
        }

        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto jacobian = qp_jacobian_(j);
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto dphi = shape_deriv_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_(i, k) * dphi / jacobian;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_prime_(j, k) = local_total[k];
            }
        }

        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 4>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 4; ++k) {
                    local_total[k] += node_u_(i, k + 3) * phi;
                }
            }
            const auto length = Kokkos::sqrt(
                local_total[0] * local_total[0] + local_total[1] * local_total[1] +
                local_total[2] * local_total[2] + local_total[3] * local_total[3]
            );
            static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
            if (length == 0.) {
                local_total = length_zero_result;
            }
            for (int k = 0; k < 4; ++k) {
                qp_r_(j, k) = local_total[k];
            }
        }

        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto jacobian = qp_jacobian_(j);
            auto local_total = Kokkos::Array<double, 4>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto dphi = shape_deriv_(i, j_index);
                for (int k = 0; k < 4; ++k) {
                    local_total[k] += node_u_(i, k + 3) * dphi / jacobian;
                }
            }
            for (int k = 0; k < 4; ++k) {
                qp_r_prime_(j, k) = local_total[k];
            }
        }
    }
};

}
