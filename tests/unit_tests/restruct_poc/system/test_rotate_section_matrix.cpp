#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/rotate_section_matrix.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(RotateSectionMatrixTests, OneNode) {
    auto rr0 = Kokkos::View<double[1][6][6]>("rr0");
    auto rr0_data = std::array<double, 36>{1., 2., 3., 0., 0., 0., 4., 5., 6., 0., 0., 0.,
                                           7., 8., 9., 0., 0., 0., 0., 0., 0., 1., 2., 3.,
                                           0., 0., 0., 4., 5., 6., 0., 0., 0., 7., 8., 9.};
    auto rr0_host = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(rr0_data.data());
    auto rr0_mirror = Kokkos::create_mirror(rr0);
    Kokkos::deep_copy(rr0_mirror, rr0_host);
    Kokkos::deep_copy(rr0, rr0_mirror);

    auto Cstar = Kokkos::View<double[1][6][6]>("Cstar");
    auto Cstar_data = std::array<double, 36>{
        1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14., 15., 16., 17., 18.,
        19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    auto Cstar_host = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(Cstar_data.data());
    auto Cstar_mirror = Kokkos::create_mirror(Cstar);
    Kokkos::deep_copy(Cstar_mirror, Cstar_host);
    Kokkos::deep_copy(Cstar, Cstar_mirror);

    auto Cuu = Kokkos::View<double[1][6][6]>("Cuu");

    Kokkos::parallel_for("RotateSectionMatrix", 1, RotateSectionMatrix{rr0, Cstar, Cuu});

    auto Cuu_exact_data = std::array<double, 36>{
        372.,  912.,  1452., 480.,  1182., 1884.,  822.,  2010., 3198.,  1092., 2685.,  4278.,
        1272., 3108., 4944., 1704., 4188., 6672.,  1020., 2532., 4044.,  1128., 2802.,  4476.,
        2442., 6060., 9678., 2712., 6735., 10758., 3864., 9588., 15312., 4296., 10668., 17040.};
    auto Cuu_exact = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(Cuu_exact_data.data());

    auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_EQ(Cuu_mirror(0, i, j), Cuu_exact(0, i, j));
        }
    }
}

}  // namespace openturbine::restruct_poc::tests