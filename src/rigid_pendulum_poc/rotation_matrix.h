#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

class RotationMatrix {
public:
    KOKKOS_FUNCTION
    RotationMatrix() {
        data[0][0] = 0.;
        data[0][1] = 0.;
        data[0][2] = 0.;

        data[1][0] = 0.;
        data[1][1] = 0.;
        data[1][2] = 0.;

        data[2][0] = 0.;
        data[2][1] = 0.;
        data[2][2] = 0.;
    }

    KOKKOS_FUNCTION
    RotationMatrix(
        double d00, double d01, double d02, double d10, double d11, double d12, double d20,
        double d21, double d22
    ) {
        data[0][0] = d00;
        data[0][1] = d01;
        data[0][2] = d02;

        data[1][0] = d10;
        data[1][1] = d11;
        data[1][2] = d12;

        data[2][0] = d20;
        data[2][1] = d21;
        data[2][2] = d22;
    }

    KOKKOS_FUNCTION
    const double& operator()(int i, int j) const { return data[i][j]; }

    KOKKOS_FUNCTION
    double& operator()(int i, int j) { return data[i][j]; }
    double data[3][3];
};

}  // namespace openturbine::rigid_pendulum