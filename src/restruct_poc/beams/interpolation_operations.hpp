#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

KOKKOS_INLINE_FUNCTION
void InterpVector3(
    const View_NxN::const_type& shape_matrix, const View_Nx3::const_type& node_v,
    const View_Nx3& qp_v
) {
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i = 0; i < node_v.extent_int(0); ++i) {
            const auto phi = shape_matrix(i, j);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_v(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_v(j, k) = local_total[k];
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector4(
    const View_NxN::const_type& shape_matrix, const View_Nx4::const_type& node_v,
    const View_Nx4& qp_v
) {
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        auto local_total = Kokkos::Array<double, 4>{};
        for (int i = 0; i < node_v.extent_int(0); ++i) {
            const auto phi = shape_matrix(i, j);
            for (int k = 0; k < 4; ++k) {
                local_total[k] += node_v(i, k) * phi;
            }
        }
        for (int k = 0; k < 4; ++k) {
            qp_v(j, k) = local_total[k];
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpQuaternion(
    const View_NxN::const_type& shape_matrix, const View_Nx4::const_type& node_v,
    const View_Nx4& qp_v
) {
    InterpVector4(shape_matrix, node_v, qp_v);
    static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        auto length = Kokkos::sqrt(
            Kokkos::pow(qp_v(j, 0), 2) + Kokkos::pow(qp_v(j, 1), 2) + Kokkos::pow(qp_v(j, 2), 2) +
            Kokkos::pow(qp_v(j, 3), 2)
        );
        if (length == 0.) {
            for (int k = 0; k < 4; ++k) {
                qp_v(j, k) = length_zero_result[k];
            }
        } else {
            for (int k = 0; k < 4; ++k) {
                qp_v(j, k) /= length;
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector3Deriv(
    const View_NxN::const_type& shape_matrix_deriv, const View_N::const_type& jacobian,
    const View_Nx3::const_type& node_v, const View_Nx3& qp_v
) {
    InterpVector3(shape_matrix_deriv, node_v, qp_v);
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        const auto jac = jacobian(j);
        for (int k = 0; k < qp_v.extent_int(1); ++k) {
            qp_v(j, k) /= jac;
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector4Deriv(
    const View_NxN::const_type& shape_matrix_deriv, const View_N::const_type& jacobian,
    const View_Nx4::const_type& node_v, const View_Nx4& qp_v
) {
    InterpVector4(shape_matrix_deriv, node_v, qp_v);
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        const auto jac = jacobian(j);
        for (int k = 0; k < qp_v.extent_int(1); ++k) {
            qp_v(j, k) /= jac;
        }
    }
}

}  // namespace openturbine
