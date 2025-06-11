#include <stddef.h>

#include <array>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <gtest/gtest.h>

#include "system/beams/rotate_section_matrix.hpp"
#include "test_calculate.hpp"

namespace {

void TestRotateSectionMatrix() {
    const auto rr0 = openturbine::tests::CreateView<double[4]>("rr0", std::array{1., 2., 3., 4.});
    const auto Cstar = openturbine::tests::CreateView<double[6][6]>(
        "Cstar", std::array{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                            13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
                            25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.}
    );

    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");

    Kokkos::parallel_for(
        "RotateSectionMatrix", 1,
        KOKKOS_LAMBDA(size_t) { openturbine::beams::RotateSectionMatrix(rr0, Cstar, Cuu); }
    );

    constexpr auto Cuu_exact_data =
        std::array{2052., 9000.,  12564., 2160., 9540.,  13320., 2700., 7200.,  9900.,
                   3240., 9900.,  13680., 3564., 9000.,  12348., 4320., 12780., 17640.,
                   2700., 12240., 17100., 2808., 12780., 17856., 5940., 23400., 32580.,
                   6480., 26100., 36360., 8100., 31680., 44100., 8856., 35460., 49392.};
    const auto Cuu_exact =
        Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type(Cuu_exact_data.data());

    const auto Cuu_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Cuu);
    openturbine::tests::CompareWithExpected(Cuu_mirror, Cuu_exact);
}

}  // namespace

namespace openturbine::tests {

TEST(RotateSectionMatrixTests, OneNode) {
    TestRotateSectionMatrix();
}

}  // namespace openturbine::tests
