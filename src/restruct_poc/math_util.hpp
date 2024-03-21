#pragma once

#include "types.hpp"

namespace oturb {

KOKKOS_FUNCTION void InterpVector3(View_NxN shape_matrix, View_Nx3 node_v, View_Nx3 qp_v);
KOKKOS_FUNCTION void InterpVector4(View_NxN shape_matrix, View_Nx4 node_v, View_Nx4 qp_v);
KOKKOS_FUNCTION void InterpQuaternion(View_NxN shape_matrix, View_Nx4 node_v, View_Nx4 qp_v);
KOKKOS_FUNCTION void InterpVector3Deriv(
    View_NxN shape_matrix, View_N jacobian, View_Nx3 node_v, View_Nx3 qp_v
);
KOKKOS_FUNCTION void InterpVector4Deriv(
    View_NxN shape_matrix, View_N jacobian, View_Nx4 node_v, View_Nx4 qp_v
);

}