#pragma once

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace openturbine::tests {

constexpr double kTolerance = 1e-15;

inline void CompareWithExpected(
    const Kokkos::View<const double*>::host_mirror_type& result,
    const Kokkos::View<const double*, Kokkos::HostSpace>& expected,
    const double tolerance = kTolerance
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        EXPECT_NEAR(result(i), expected(i), tolerance);
    }
}

inline void CompareWithExpected(
    const Kokkos::View<const double**>::host_mirror_type& result,
    const Kokkos::View<const double**, Kokkos::HostSpace>& expected,
    const double tolerance = kTolerance
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            EXPECT_NEAR(result(i, j), expected(i, j), tolerance);
        }
    }
}

inline void CompareWithExpected(
    const Kokkos::View<const double***>::host_mirror_type& result,
    const Kokkos::View<const double***, Kokkos::HostSpace>& expected,
    const double tolerance = kTolerance
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            for (auto k = 0U; k < result.extent(2); ++k) {
                EXPECT_NEAR(result(i, j, k), expected(i, j, k), tolerance);
            }
        }
    }
}

inline void CompareWithExpected(
    const Kokkos::View<const double****>::host_mirror_type& result,
    const Kokkos::View<const double****, Kokkos::HostSpace>& expected,
    const double tolerance = kTolerance
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            for (auto k = 0U; k < result.extent(2); ++k) {
                for (auto l = 0U; l < result.extent(3); ++l) {
                    EXPECT_NEAR(result(i, j, k, l), expected(i, j, k, l), tolerance);
                }
            }
        }
    }
}

inline void CompareWithExpected(
    const Kokkos::View<const double*****>::host_mirror_type& result,
    const Kokkos::View<const double*****, Kokkos::HostSpace>& expected,
    const double tolerance = kTolerance
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        for (auto j = 0U; j < result.extent(1); ++j) {
            for (auto k = 0U; k < result.extent(2); ++k) {
                for (auto l = 0U; l < result.extent(3); ++l) {
                    for (auto m = 0U; m < result.extent(4); ++m) {
                        EXPECT_NEAR(result(i, j, k, l, m), expected(i, j, k, l, m), tolerance);
                    }
                }
            }
        }
    }
}

}  // namespace openturbine::tests
