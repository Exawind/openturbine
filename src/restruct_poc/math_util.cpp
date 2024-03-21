#include "math_util.hpp"

namespace oturb {

KOKKOS_FUNCTION
void InterpVector3(View_NxN shape_matrix, View_Nx3 node_v, View_Nx3 qp_v) {
    Kokkos::deep_copy(qp_v, 0.);
    for (size_t i = 0; i < node_v.extent(0); ++i) {
        for (size_t j = 0; j < qp_v.extent(0); ++j) {
            for (size_t k = 0; k < 3; ++k) {
                qp_v(j, k) += node_v(i, k) * shape_matrix(i, j);
            }
        }
    }
}

KOKKOS_FUNCTION
void InterpVector4(View_NxN shape_matrix, View_Nx4 node_v, View_Nx4 qp_v) {
    Kokkos::deep_copy(qp_v, 0.);
    for (size_t i = 0; i < node_v.extent(0); ++i) {
        for (size_t j = 0; j < qp_v.extent(0); ++j) {
            for (size_t k = 0; k < 4; ++k) {
                qp_v(j, k) += node_v(i, k) * shape_matrix(i, j);
            }
        }
    }
}

KOKKOS_FUNCTION
void InterpQuaternion(View_NxN shape_matrix, View_Nx4 node_v, View_Nx4 qp_v) {
    InterpVector4(shape_matrix, node_v, qp_v);

    // Normalize quaternions (rows)
    for (size_t j = 0; j < qp_v.extent(0); ++j) {
        auto length = Kokkos::sqrt(
            Kokkos::pow(qp_v(j, 0), 2) + Kokkos::pow(qp_v(j, 1), 2) + Kokkos::pow(qp_v(j, 2), 2) +
            Kokkos::pow(qp_v(j, 3), 2)
        );
        if (length == 0.) {
            qp_v(j, 0) = 1.;
            qp_v(j, 3) = 0.;
            qp_v(j, 2) = 0.;
            qp_v(j, 1) = 0.;
        } else {
            qp_v(j, 0) /= length;
            qp_v(j, 3) /= length;
            qp_v(j, 2) /= length;
            qp_v(j, 1) /= length;
        }
    }
}

KOKKOS_FUNCTION
void InterpVector3Deriv(
    View_NxN shape_matrix_deriv, View_N jacobian, View_Nx3 node_v, View_Nx3 qp_v
) {
    InterpVector3(shape_matrix_deriv, node_v, qp_v);
    for (size_t j = 0; j < qp_v.extent(0); ++j) {
        for (size_t k = 0; k < qp_v.extent(1); ++k) {
            qp_v(j, k) /= jacobian(j);
        }
    }
}

KOKKOS_FUNCTION
void InterpVector4Deriv(
    View_NxN shape_matrix_deriv, View_N jacobian, View_Nx4 node_v, View_Nx4 qp_v
) {
    InterpVector4(shape_matrix_deriv, node_v, qp_v);
    for (size_t j = 0; j < qp_v.extent(0); ++j) {
        for (size_t k = 0; k < qp_v.extent(1); ++k) {
            qp_v(j, k) /= jacobian(j);
        }
    }
}
}  // namespace oturb