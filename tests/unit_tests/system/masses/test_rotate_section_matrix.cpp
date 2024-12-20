#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"

#include "src/system/masses/rotate_section_matrix.hpp"

namespace openturbine::tests {

struct ExecuteRotateSectionMatrix {
    size_t i_elem;
    Kokkos::View<double[1][6][6]>::const_type rr0;
    Kokkos::View<double[1][6][6]>::const_type Cstar;
    Kokkos::View<double[1][6][6]> Cuu;

    KOKKOS_FUNCTION
    void operator()(size_t) const { masses::RotateSectionMatrix{i_elem, rr0, Cstar, Cuu}(); };
};

TEST(RotateSectionMatrixMassesTests, OneNode) {
    const auto rr0 = Kokkos::View<double[1][6][6]>("rr0");
    constexpr auto rr0_data =
        std::array{1., 2., 3., 0., 0., 0., 4., 5., 6., 0., 0., 0., 7., 8., 9., 0., 0., 0.,
                   0., 0., 0., 1., 2., 3., 0., 0., 0., 4., 5., 6., 0., 0., 0., 7., 8., 9.};
    const auto rr0_host = Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(rr0_data.data());
    const auto rr0_mirror = Kokkos::create_mirror(rr0);
    Kokkos::deep_copy(rr0_mirror, rr0_host);
    Kokkos::deep_copy(rr0, rr0_mirror);

    const auto Cstar = Kokkos::View<double[1][6][6]>("Cstar");
    constexpr auto Cstar_data =
        std::array{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                   13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
                   25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    const auto Cstar_host =
        Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(Cstar_data.data());
    const auto Cstar_mirror = Kokkos::create_mirror(Cstar);
    Kokkos::deep_copy(Cstar_mirror, Cstar_host);
    Kokkos::deep_copy(Cstar, Cstar_mirror);

    const auto Cuu = Kokkos::View<double[1][6][6]>("Cuu");

    Kokkos::parallel_for("RotateSectionMatrix", 1, ExecuteRotateSectionMatrix{0, rr0, Cstar, Cuu});

    constexpr auto Cuu_exact_data =
        std::array{372.,  912.,  1452.,  480.,  1182., 1884.,  822.,  2010.,  3198.,
                   1092., 2685., 4278.,  1272., 3108., 4944.,  1704., 4188.,  6672.,
                   1020., 2532., 4044.,  1128., 2802., 4476.,  2442., 6060.,  9678.,
                   2712., 6735., 10758., 3864., 9588., 15312., 4296., 10668., 17040.};
    const auto Cuu_exact =
        Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(Cuu_exact_data.data());

    const auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu);
    CompareWithExpected(Cuu_mirror, Cuu_exact);
}

}  // namespace openturbine::tests