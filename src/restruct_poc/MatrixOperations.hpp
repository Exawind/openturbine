#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename View_A>
KOKKOS_INLINE_FUNCTION void FillMatrix(View_A A, double value) {
    for (int i = 0; i < A.extent_int(0); ++i) {
        for (int j = 0; j < A.extent_int(1); ++j) {
            A(i, j) = value;
        }
    }
}

template <typename A, typename B>
KOKKOS_INLINE_FUNCTION void MatScale(A m_in, double scale, B m_out) {
    for (int i = 0; i < m_in.extent_int(0); ++i) {
        for (int j = 0; j < m_in.extent_int(1); ++j) {
            m_out(i, j) = m_in(i, j) * scale;
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatAdd(View_A M_A, View_B M_B, View_C M_C) {
    for (int i = 0; i < M_A.extent_int(0); ++i) {
        for (int j = 0; j < M_A.extent_int(1); ++j) {
            M_C(i, j) = M_A(i, j) + M_B(i, j);
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulAB(View_A A, View_B B, View_C C) {
    for (int i = 0; i < A.extent_int(0); ++i) {
        for (int j = 0; j < B.extent_int(1); ++j) {
            auto local_result = 0.;
            for (int k = 0; k < B.extent_int(0); ++k) {
                local_result += A(i, k) * B(k, j);
            }
            C(i, j) = local_result;
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulATB(View_A AT, View_B B, View_C C) {
    for (int i = 0; i < AT.extent_int(1); ++i) {
        for (int j = 0; j < B.extent_int(1); ++j) {
            auto local_result = 0.;
            for (int k = 0; k < B.extent_int(0); ++k) {
                local_result += AT(k, i) * B(k, j);
            }
            C(i, j) = local_result;
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulABT(View_A A, View_B BT, View_C C) {
    for (int i = 0; i < A.extent_int(0); ++i) {
        for (int j = 0; j < BT.extent_int(0); ++j) {
            auto local_result = 0.;
            for (int k = 0; k < BT.extent_int(1); ++k) {
                local_result += A(i, k) * BT(j, k);
            }
            C(i, j) = local_result;
        }
    }
}

}
