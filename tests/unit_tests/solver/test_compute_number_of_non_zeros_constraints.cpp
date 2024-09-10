#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/compute_number_of_non_zeros_constraints.hpp"

namespace openturbine::tests {

TEST(ComputeNumberOfNonZeros_Constraints, OneOfEach) {
    constexpr auto num_constraints = 5U;
    constexpr auto type_host_data = std::array{
        ConstraintType::kFixedBC, ConstraintType::kPrescribedBC, ConstraintType::kRigid,
        ConstraintType::kCylindrical, ConstraintType::kRotationControl};
    constexpr auto row_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{0U, 6U}, Kokkos::pair<size_t, size_t>{6U, 12U},
        Kokkos::pair<size_t, size_t>{12U, 18U}, Kokkos::pair<size_t, size_t>{18U, 24U},
        Kokkos::pair<size_t, size_t>{24U, 30U}};

    const auto type_host =
        Kokkos::View<const ConstraintType[num_constraints], Kokkos::HostSpace>(type_host_data.data()
        );
    const auto type = Kokkos::View<ConstraintType[num_constraints]>("type");
    Kokkos::deep_copy(type, type_host);

    const auto row_range_host =
        Kokkos::View<const Kokkos::pair<size_t, size_t>[num_constraints], Kokkos::HostSpace>(
            row_range_host_data.data()
        );
    const auto row_range = Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints]>("row_range");
    Kokkos::deep_copy(row_range, row_range_host);

    auto nnz = size_t{0U};
    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros_Constraints", num_constraints,
        ComputeNumberOfNonZeros_Constraints{type, row_range}, nnz
    );

    EXPECT_EQ(nnz, 288U);
}

}  // namespace openturbine::tests
