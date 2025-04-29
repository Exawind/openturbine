#pragma once

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace openturbine::tests {

template <typename ValueType, typename DataType>
typename Kokkos::View<ValueType>::const_type CreateView(
    const std::string& name, const DataType& data
) {
    const auto view = Kokkos::View<ValueType>(name);
    const auto host = typename Kokkos::View<ValueType, Kokkos::HostSpace>::const_type(data.data());
    const auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, host);
    Kokkos::deep_copy(view, mirror);
    return view;
}

inline void CompareWithExpected(
    const Kokkos::View<const double*>::host_mirror_type& result,
    const Kokkos::View<const double*, Kokkos::HostSpace>& expected
) {
    for (auto i = 0U; i < result.extent(0); ++i) {
        EXPECT_DOUBLE_EQ(result(i), expected(i));
    }
}

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
}  // namespace openturbine::tests
