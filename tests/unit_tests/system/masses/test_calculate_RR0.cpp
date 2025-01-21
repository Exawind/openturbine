#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/masses/calculate_RR0.hpp"
#include "test_calculate.hpp"

namespace openturbine::tests {

struct ExecuteCalculateRR0 {
    size_t i_elem;
    Kokkos::View<double[1][7]>::const_type x;
    Kokkos::View<double[1][6][6]> rr0;

    KOKKOS_FUNCTION
    void operator()(size_t) const { masses::CalculateRR0{i_elem, x, rr0}(); }
};

TEST(CalculateRR0MassesTests, OneNode) {
    const auto x = Kokkos::View<double[1][7]>("x");
    constexpr auto x_host_data = std::array{1., 2., 3., 4., 5., 6., 7.};
    const auto x_host = Kokkos::View<const double[1][7], Kokkos::HostSpace>(x_host_data.data());
    const auto x_mirror = Kokkos::create_mirror(x);
    Kokkos::deep_copy(x_mirror, x_host);
    Kokkos::deep_copy(x, x_mirror);

    const auto rr0 = Kokkos::View<double[1][6][6]>("rr0");

    Kokkos::parallel_for("CalculateRR0", 1, ExecuteCalculateRR0{0, x, rr0});

    constexpr auto expected_rr0_data = std::array{-44., 4.,   118., 0.,   0.,   0.,    //
                                                  116., -22., 44.,  0.,   0.,   0.,    //
                                                  22.,  124., 4.,   0.,   0.,   0.,    //
                                                  0.,   0.,   0.,   -44., 4.,   118.,  //
                                                  0.,   0.,   0.,   116., -22., 44.,   //
                                                  0.,   0.,   0.,   22.,  124., 4.};
    const auto expected_rr0 =
        Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(expected_rr0_data.data());

    const auto rr0_mirror = Kokkos::create_mirror(rr0);
    Kokkos::deep_copy(rr0_mirror, rr0);
    CompareWithExpected(rr0_mirror, expected_rr0);
}
}  // namespace openturbine::tests
