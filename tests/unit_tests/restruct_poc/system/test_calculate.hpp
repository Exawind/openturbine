#pragma once

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace openturbine::tests {

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

inline void CompareWithExpected(
    const Kokkos::View<const double****>::host_mirror_type& result,
    const Kokkos::View<const double****, Kokkos::HostSpace>& expected
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            for (auto k = 0U; k < result.extent(2); ++k) {
                for (auto l = 0U; l < result.extent(3); ++l) {
                    EXPECT_DOUBLE_EQ(result(i, j, k, l), expected(i, j, k, l));
                }
            }
        }
    }
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
inline void CompareWithExpected(
    const Kokkos::View<const double*****>::host_mirror_type& result,
    const Kokkos::View<const double*****, Kokkos::HostSpace>& expected
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            for (auto k = 0U; k < result.extent(2); ++k) {
                for (auto l = 0U; l < result.extent(3); ++l) {
                    for (auto m = 0U; m < result.extent(4); ++m) {
                        EXPECT_DOUBLE_EQ(result(i, j, k, l, m), expected(i, j, k, l, m));
                    }
                }
            }
        }
    }
}
// NOLINTEND(readability-function-cognitive-complexity)
}  // namespace openturbine::tests
