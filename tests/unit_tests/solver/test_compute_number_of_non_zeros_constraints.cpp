#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "solver/compute_number_of_non_zeros_constraints.hpp"

namespace openturbine::tests {

TEST(ComputeNumberOfNonZeros_Constraints, OneOfEach) {
    constexpr auto num_constraints = 5U;
    constexpr auto row_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{0U, 6U}, Kokkos::pair<size_t, size_t>{6U, 12U},
        Kokkos::pair<size_t, size_t>{12U, 18U}, Kokkos::pair<size_t, size_t>{18U, 24U},
        Kokkos::pair<size_t, size_t>{24U, 30U}
    };
    const auto base_node_freedom_signature_host_data = std::array{
        FreedomSignature::NoComponents, FreedomSignature::NoComponents,
        FreedomSignature::AllComponents, FreedomSignature::AllComponents,
        FreedomSignature::AllComponents
    };
    const auto target_node_freedom_signature_host_data = std::array{
        FreedomSignature::AllComponents, FreedomSignature::AllComponents,
        FreedomSignature::AllComponents, FreedomSignature::AllComponents,
        FreedomSignature::AllComponents
    };

    const auto row_range_host =
        Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints], Kokkos::HostSpace>::const_type(
            row_range_host_data.data()
        );
    const auto row_range = Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints]>("row_range");
    Kokkos::deep_copy(row_range, row_range_host);

    const auto base_node_freedom_signature_host =
        Kokkos::View<FreedomSignature[num_constraints], Kokkos::HostSpace>::const_type(
            base_node_freedom_signature_host_data.data()
        );
    const auto target_node_freedom_signature_host =
        Kokkos::View<FreedomSignature[num_constraints], Kokkos::HostSpace>::const_type(
            target_node_freedom_signature_host_data.data()
        );

    const auto base_node_freedom_signature =
        Kokkos::View<FreedomSignature[num_constraints]>("base_nfs");
    const auto target_node_freedom_signature =
        Kokkos::View<FreedomSignature[num_constraints]>("target_nfs");

    Kokkos::deep_copy(base_node_freedom_signature, base_node_freedom_signature_host);
    Kokkos::deep_copy(target_node_freedom_signature, target_node_freedom_signature_host);

    auto nnz = size_t{0U};
    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros_Constraints", num_constraints,
        ComputeNumberOfNonZeros_Constraints{
            row_range, base_node_freedom_signature, target_node_freedom_signature
        },
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
    constexpr auto row_range_host_data = std::array{
        Kokkos::pair<size_t, size_t>{0U, 3U}, Kokkos::pair<size_t, size_t>{3U, 6U},
        Kokkos::pair<size_t, size_t>{6U, 9U}
    };
    const auto base_node_freedom_signature_host_data = std::array{
        FreedomSignature::NoComponents, FreedomSignature::NoComponents,
        FreedomSignature::AllComponents
    };
    const auto target_node_freedom_signature_host_data = std::array{
        FreedomSignature::JustPosition, FreedomSignature::JustPosition,
        FreedomSignature::JustPosition
    };

    const auto row_range_host =
        Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints], Kokkos::HostSpace>::const_type(
            row_range_host_data.data()
        );
    const auto row_range = Kokkos::View<Kokkos::pair<size_t, size_t>[num_constraints]>("row_range");
    Kokkos::deep_copy(row_range, row_range_host);

    const auto base_node_freedom_signature_host =
        Kokkos::View<FreedomSignature[num_constraints], Kokkos::HostSpace>::const_type(
            base_node_freedom_signature_host_data.data()
        );
    const auto target_node_freedom_signature_host =
        Kokkos::View<FreedomSignature[num_constraints], Kokkos::HostSpace>::const_type(
            target_node_freedom_signature_host_data.data()
        );

    const auto base_node_freedom_signature =
        Kokkos::View<FreedomSignature[num_constraints]>("base_nfs");
    const auto target_node_freedom_signature =
        Kokkos::View<FreedomSignature[num_constraints]>("target_nfs");

    Kokkos::deep_copy(base_node_freedom_signature, base_node_freedom_signature_host);
    Kokkos::deep_copy(target_node_freedom_signature, target_node_freedom_signature_host);

    auto nnz = size_t{0U};
    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros_Constraints", num_constraints,
        ComputeNumberOfNonZeros_Constraints{
            row_range, base_node_freedom_signature, target_node_freedom_signature
        },
        nnz
    );

    // Expected values:
    // kFixedBC3DOFs: 3 rows * 3 cols = 9
    // kPrescribedBC3DOFs: 3 rows * 3 cols = 9
    // kRigidJoint6DOFsTo3DOFs: 3 rows * 9 cols = 27
    EXPECT_EQ(nnz, 9U + 9U + 27U);
}

}  // namespace openturbine::tests
