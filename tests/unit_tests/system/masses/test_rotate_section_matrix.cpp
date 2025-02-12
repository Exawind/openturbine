#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/masses/rotate_section_matrix.hpp"
#include "test_calculate.hpp"

namespace openturbine::tests {

struct ExecuteRotateSectionMatrix {
    Kokkos::View<double[4]>::const_type xr;
    Kokkos::View<double[6][6]>::const_type Cstar;
    Kokkos::View<double[6][6]> Cuu;

    KOKKOS_FUNCTION
    void operator()(size_t) const { masses::RotateSectionMatrix(xr, Cstar, Cuu); };
};

TEST(RotateSectionMatrixMassesTests, OneNode) {
    const auto xr = Kokkos::View<double[4]>("xr");
    constexpr auto xr_data = std::array{1., 2., 3., 4.};
    const auto xr_host = Kokkos::View<const double[4], Kokkos::HostSpace>(xr_data.data());
    const auto xr_mirror = Kokkos::create_mirror(xr);
    Kokkos::deep_copy(xr_mirror, xr_host);
    Kokkos::deep_copy(xr, xr_mirror);

    const auto Cstar = Kokkos::View<double[6][6]>("Cstar");
    constexpr auto Cstar_data =
        std::array{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                   13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
                   25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    const auto Cstar_host =
        Kokkos::View<const double[6][6], Kokkos::HostSpace>(Cstar_data.data());
    const auto Cstar_mirror = Kokkos::create_mirror(Cstar);
    Kokkos::deep_copy(Cstar_mirror, Cstar_host);
    Kokkos::deep_copy(Cstar, Cstar_mirror);

    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");

    Kokkos::parallel_for("RotateSectionMatrix", 1, ExecuteRotateSectionMatrix{xr, Cstar, Cuu});

    constexpr auto Cuu_exact_data =
        std::array{2052., 9000.,  12564., 2160., 9540.,  13320., 2700., 7200.,  9900.,
                   3240., 9900.,  13680., 3564., 9000.,  12348., 4320., 12780., 17640.,
                   2700., 12240., 17100., 2808., 12780., 17856., 5940., 23400., 32580.,
                   6480., 26100., 36360., 8100., 31680., 44100., 8856., 35460., 49392.};
    const auto Cuu_exact =
        Kokkos::View<const double[6][6], Kokkos::HostSpace>(Cuu_exact_data.data());

    const auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu);
    CompareWithExpected(Cuu_mirror, Cuu_exact);
}

}  // namespace openturbine::tests
