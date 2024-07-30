#pragma once

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace openturbine::restruct_poc::tests {

inline void CompareWithExpected(
    const Kokkos::View<const double**>::host_mirror_type& result,
    const Kokkos::View<const double**, Kokkos::HostSpace>& expected
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            EXPECT_DOUBLE_EQ(result(i, j), expected(i, j));
        }
    }
}

inline void CompareWithExpected(
    const Kokkos::View<const double***>::host_mirror_type& result,
    const Kokkos::View<const double***, Kokkos::HostSpace>& expected
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            for (auto k = 0U; k < result.extent(2); ++k) {
                EXPECT_DOUBLE_EQ(result(i, j, k), expected(i, j, k));
            }
        }
    }
}

}  // namespace openturbine::restruct_poc::tests