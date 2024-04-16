#pragma once
#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename View_A>
KOKKOS_INLINE_FUNCTION void FillVector(View_A A, double value) {
    for (int i = 0; i < A.extent_int(0); ++i) {
        A(i) = value;
    }    
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatVecMulAB(View_A A, View_B B, View_C C) { 
    for (int i = 0; i < A.extent_int(0); ++i) {
        auto local_result = 0.;
        for (int k = 0; k < B.extent_int(0); ++k) {
            local_result += A(i, k) * B(k);
        }
        C(i) = local_result;
    }    
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatVecMulATB(View_A A, View_B B, View_C C) { 
    for (int i = 0; i < A.extent_int(1); ++i) {
        auto local_result = 0.;
        for (int k = 0; k < B.extent_int(0); ++k) {
            local_result += A(k, i) * B(k);
        }
        C(i) = local_result;
    }    
}

template <typename VectorType, typename MatrixType>
KOKKOS_INLINE_FUNCTION void VecTilde(VectorType vector, MatrixType matrix) {
    matrix(0, 0) = 0.;
    matrix(0, 1) = -vector(2);
    matrix(0, 2) = vector(1);
    matrix(1, 0) = vector(2);
    matrix(1, 1) = 0.;
    matrix(1, 2) = -vector(0);
    matrix(2, 0) = -vector(1);
    matrix(2, 1) = vector(0);
    matrix(2, 2) = 0.;
}

template <typename V1, typename V2>
KOKKOS_INLINE_FUNCTION void VecScale(V1 v_in, double scale, V2 v_out) {
    for (int i = 0; i < v_in.extent_int(0); ++i) {
        v_out(i) = v_in(i) * scale;
    }
}

}
