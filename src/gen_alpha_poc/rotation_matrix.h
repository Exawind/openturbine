#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::gen_alpha_solver {

class RotationMatrix {
public:
    KOKKOS_FUNCTION
    RotationMatrix(
        double d00 = 0., double d01 = 0., double d02 = 0., double d10 = 0., double d11 = 0.,
        double d12 = 0., double d20 = 0., double d21 = 0., double d22 = 0.
    ) {
        data_[0][0] = d00;
        data_[0][1] = d01;
        data_[0][2] = d02;

        data_[1][0] = d10;
        data_[1][1] = d11;
        data_[1][2] = d12;

        data_[2][0] = d20;
        data_[2][1] = d21;
        data_[2][2] = d22;
    }

    KOKKOS_FUNCTION
    const double& operator()(int i, int j) const { return data_[i][j]; }

    KOKKOS_FUNCTION
    double& operator()(int i, int j) { return data_[i][j]; }

private:
    double data_[3][3];
};

}  // namespace openturbine::gen_alpha_solver
