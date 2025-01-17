#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/compute_number_of_non_zeros_constraints.hpp"

namespace openturbine::tests {

TEST(ComputeNumberOfNonZeros_Constraints, OneOfEach) {
    constexpr auto num_constraints = 5U;
    constexpr auto type_host_data = std::array{
        ConstraintType::kFixedBC, ConstraintType::kPrescribedBC, ConstraintType::kRigidJoint,
        ConstraintType::kRevoluteJoint, ConstraintType::kRotationControl
    };
    constexpr auto row_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{0U, 6U}, Kokkos::pair<size_t, size_t>{6U, 12U},
        Kokkos::pair<size_t, size_t>{12U, 18U}, Kokkos::pair<size_t, size_t>{18U, 24U},
        Kokkos::pair<size_t, size_t>{24U, 30U}
    };
    const auto base_node_col_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{0U, 0U}, Kokkos::pair<size_t, size_t>{0U, 0U},
        Kokkos::pair<size_t, size_t>{0U, 6U}, Kokkos::pair<size_t, size_t>{0U, 6U},
        Kokkos::pair<size_t, size_t>{0U, 6U}
    };
    const auto target_node_col_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{6U, 12U}, Kokkos::pair<size_t, size_t>{6U, 12U},
        Kokkos::pair<size_t, size_t>{6U, 12U}, Kokkos::pair<size_t, size_t>{6U, 12U},
        Kokkos::pair<size_t, size_t>{6U, 12U}
    };

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

    const auto base_node_col_range_host =
        Kokkos::View<const Kokkos::pair<size_t, size_t>[num_constraints], Kokkos::HostSpace>(
            base_node_col_range_host_data.data()
        );
    const auto base_node_col_range =
        Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints]>("base_node_col_range");
    Kokkos::deep_copy(base_node_col_range, base_node_col_range_host);

    const auto target_node_col_range_host =
        Kokkos::View<const Kokkos::pair<size_t, size_t>[num_constraints], Kokkos::HostSpace>(
            target_node_col_range_host_data.data()
        );
    const auto target_node_col_range =
        Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints]>("target_node_col_range");
    Kokkos::deep_copy(target_node_col_range, target_node_col_range_host);

    auto nnz = size_t{0U};
    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros_Constraints", num_constraints,
        ComputeNumberOfNonZeros_Constraints{row_range, base_node_col_range, target_node_col_range},
        nnz
    );

    // Expected values:
    // kFixedBC: 6 rows * 6 cols = 36
    // kPrescribedBC: 6 rows * 6 cols = 36
    // kRigidJoint: 6 rows * 12 cols = 72
    // kRevoluteJoint: 6 rows * 12 cols = 72
    // kRotationControl: 6 rows * 12 cols = 72
    EXPECT_EQ(nnz, 36U + 36U + 72U + 72U + 72U);
}

TEST(ComputeNumberOfNonZeros_Constraints, ThreeDOFConstraints) {
    constexpr auto num_constraints = 3U;
    constexpr auto type_host_data = std::array{
        ConstraintType::kFixedBC3DOFs, ConstraintType::kPrescribedBC3DOFs,
        ConstraintType::kRigidJoint6DOFsTo3DOFs
    };
    constexpr auto row_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{0U, 3U}, Kokkos::pair<size_t, size_t>{3U, 6U},
        Kokkos::pair<size_t, size_t>{6U, 9U}
    };
    const auto base_node_col_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{0U, 0U}, Kokkos::pair<size_t, size_t>{0U, 0U},
        Kokkos::pair<size_t, size_t>{0U, 6U}
    };
    const auto target_node_col_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{6U, 9U}, Kokkos::pair<size_t, size_t>{6U, 9U},
        Kokkos::pair<size_t, size_t>{6U, 9U}
    };

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

    const auto base_node_col_range_host =
        Kokkos::View<const Kokkos::pair<size_t, size_t>[num_constraints], Kokkos::HostSpace>(
            base_node_col_range_host_data.data()
        );
    const auto base_node_col_range =
        Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints]>("base_node_col_range");
    Kokkos::deep_copy(base_node_col_range, base_node_col_range_host);

    const auto target_node_col_range_host =
        Kokkos::View<const Kokkos::pair<size_t, size_t>[num_constraints], Kokkos::HostSpace>(
            target_node_col_range_host_data.data()
        );
    const auto target_node_col_range =
        Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints]>("target_node_col_range");
    Kokkos::deep_copy(target_node_col_range, target_node_col_range_host);

    auto nnz = size_t{0U};
    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros_Constraints", num_constraints,
        ComputeNumberOfNonZeros_Constraints{row_range, base_node_col_range, target_node_col_range},
        nnz
    );

    // Expected values:
    // kFixedBC3DOFs: 3 rows * 3 cols = 9
    // kPrescribedBC3DOFs: 3 rows * 3 cols = 9
    // kRigidJoint6DOFsTo3DOFs: 3 rows * 9 cols = 27
    EXPECT_EQ(nnz, 9U + 9U + 27U);
}

}  // namespace openturbine::tests
