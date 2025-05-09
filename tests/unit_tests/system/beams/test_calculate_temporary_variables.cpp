#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/calculate_temporary_variables.hpp"
#include "test_calculate.hpp"

namespace {

void TestCalculateTemporaryVariables() {
    const auto x0_prime =
        openturbine::tests::CreateView<double[3]>("x0_prime", std::array{1., 2., 3.});
    const auto u_prime =
        openturbine::tests::CreateView<double[3]>("u_prime", std::array{4., 5., 6.});

    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");

    Kokkos::parallel_for(
        "CalculateTemporaryVariables", 1,
        KOKKOS_LAMBDA(size_t) {
            openturbine::beams::CalculateTemporaryVariables<Kokkos::DefaultExecutionSpace>(x0_prime, u_prime, x0pupSS);
        }
    );

    constexpr auto x0pupSS_exact_data = std::array{0., -9., 7., 9., 0., -5., -7., 5., 0.};
    const auto x0pupSS_exact =
        Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type(x0pupSS_exact_data.data());

    const auto x0pupSS_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x0pupSS);
    openturbine::tests::CompareWithExpected(x0pupSS_mirror, x0pupSS_exact);
}

}  // namespace

namespace openturbine::tests {

TEST(CalculateTemporaryVariablesTests, OneNode) {
    TestCalculateTemporaryVariables();
}

}  // namespace openturbine::tests
