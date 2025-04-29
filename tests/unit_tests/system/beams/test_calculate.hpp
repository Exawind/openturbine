#pragma once

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace openturbine::tests {

constexpr double kTolerance = 1e-15;

template <typename ValueType, typename DataType>
typename Kokkos::View<ValueType>::const_type CreateView(
    const std::string& name, const DataType& data
) {
    const auto view = Kokkos::View<ValueType>(Kokkos::view_alloc(name, Kokkos::WithoutInitializing));
    const auto host = typename Kokkos::View<ValueType, Kokkos::HostSpace>::const_type(data.data());
    const auto mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(mirror, host);
    Kokkos::deep_copy(view, mirror);
    return view;
}

template <typename ValueType, typename DataType>
typename Kokkos::View<ValueType, Kokkos::LayoutLeft>::const_type CreateLeftView(
    const std::string& name, const DataType& data
) {
    const auto view = Kokkos::View<ValueType, Kokkos::LayoutLeft>(
        Kokkos::view_alloc(name, Kokkos::WithoutInitializing)
    );
    const auto host = typename Kokkos::View<ValueType, Kokkos::HostSpace>::const_type(data.data());
    const auto mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(mirror, host);
    Kokkos::deep_copy(view, mirror);
    return view;
}

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
