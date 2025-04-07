#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/rotate_section_matrix.hpp"
#include "test_calculate.hpp"

namespace openturbine::tests {

/**
 * Test data taken from: Run5-DynamicIterationMatrix.nb -> Introduce a stiffness matrix in material
 * coordinates; See Johnson's thesis
 * Ref: https://github.com/michaelasprague/OpenTurbineTheory/tree/main/mathematica
 */
TEST(RotateSectionMatrixTests, OneNode) {
    // quaternion reprenseting the rotation
    auto xr = Kokkos::View<double[4]>("xr");
    constexpr auto xr_data = std::array{0.8628070705148, 0., 0., 0.5055333412048};
    const auto xr_host = Kokkos::View<const double[4], Kokkos::HostSpace>(xr_data.data());
    const auto xr_mirror = Kokkos::create_mirror(xr);
    Kokkos::deep_copy(xr_mirror, xr_host);
    Kokkos::deep_copy(xr, xr_mirror);

    // stiffness matrix in material coordinates
    const auto Cstar = Kokkos::View<double[6][6]>("Cstar");
    constexpr auto Cstar_data = std::array{
        1.36817e6, 0.,     0.,     0.,     0.,     0.,      // row 1
        0.,        88560., 0.,     0.,     0.,     0.,      // row 2
        0.,        0.,     38780., 0.,     0.,     0.,      // row 3
        0.,        0.,     0.,     16960., 17610., -351.,   // row 4
        0.,        0.,     0.,     17610., 59120., -370.,   // row 5
        0.,        0.,     0.,     -351.,  -370.,  141470.  // row 6
    };
    const auto Cstar_host = Kokkos::View<const double[6][6], Kokkos::HostSpace>(Cstar_data.data());
    const auto Cstar_mirror = Kokkos::create_mirror(Cstar);
    Kokkos::deep_copy(Cstar_mirror, Cstar_host);
    Kokkos::deep_copy(Cstar, Cstar_mirror);

    // stiffness matrix in global coordinates
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    Kokkos::parallel_for(
        "RotateSectionMatrix", 1,
        KOKKOS_LAMBDA(size_t) { beams::RotateSectionMatrix(xr, Cstar, Cuu); }
    );

    constexpr auto Cuu_exact_data = std::array{
        394381.5594951,
        545715.5847999,
        0.,
        0.,
        0.,
        0.,  // row 1
        545715.5847999,
        1.062348440505e6,
        0.,
        0.,
        0.,
        0.,  // row 2
        0.,
        0.,
        38780.,
        0.,
        0.,
        0.,  // row 3
        0.,
        0.,
        0.,
        34023.65045212,
        -27172.54931561,
        151.1774277346,  // row 4
        0.,
        0.,
        0.,
        -27172.54931561,
        42056.34954788,
        -487.0794445915,  // row 5
        0.,
        0.,
        0.,
        151.1774277346,
        -487.0794445915,
        141470.  // row 6
    };
    const auto Cuu_exact =
        Kokkos::View<const double[6][6], Kokkos::HostSpace>(Cuu_exact_data.data());
    const auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu);
    CompareWithExpected(Cuu_mirror, Cuu_exact, 1e-6);
}

}  // namespace openturbine::tests
