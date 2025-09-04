#pragma once
#include <ranges>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace openturbine::beams::tests {

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
    for (auto i : std::views::iota(0U, result.extent(0))) {
        EXPECT_NEAR(result(i), expected(i), tolerance);
    }
}

inline void CompareWithExpected(
    const Kokkos::View<const double**>::host_mirror_type& result,
    const Kokkos::View<const double**, Kokkos::HostSpace>& expected,
    const double tolerance = kTolerance
) {
    for (auto i : std::views::iota(0U, result.extent(0))) {
        for (auto j : std::views::iota(0U, result.extent(1))) {
            EXPECT_NEAR(result(i, j), expected(i, j), tolerance);
        }
    }
}

inline void CompareWithExpected(
    const Kokkos::View<const double***>::host_mirror_type& result,
    const Kokkos::View<const double***, Kokkos::HostSpace>& expected,
    const double tolerance = kTolerance
) {
    for (auto i : std::views::iota(0U, result.extent(0))) {
        for (auto j : std::views::iota(0U, result.extent(1))) {
            for (auto k : std::views::iota(0U, result.extent(2))) {
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
    for (auto i : std::views::iota(0U, result.extent(0))) {
        for (auto j : std::views::iota(0U, result.extent(1))) {
            for (auto k : std::views::iota(0U, result.extent(2))) {
                for (auto l : std::views::iota(0U, result.extent(3))) {
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
    for (auto i : std::views::iota(0U, result.extent(0))) {
        for (auto j : std::views::iota(0U, result.extent(1))) {
            for (auto k : std::views::iota(0U, result.extent(2))) {
                for (auto l : std::views::iota(0U, result.extent(3))) {
                    for (auto m : std::views::iota(0U, result.extent(4))) {
                        EXPECT_NEAR(result(i, j, k, l, m), expected(i, j, k, l, m), tolerance);
                    }
                }
            }
        }
    }
}

}  // namespace openturbine::beams::tests
