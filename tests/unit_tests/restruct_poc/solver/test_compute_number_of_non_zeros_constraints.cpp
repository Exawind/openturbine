#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/solver/compute_number_of_non_zeros_constraints.hpp"

namespace openturbine::tests {

TEST(ComputeNumberOfNonZeros_Constraints, OneOfEach) {
    constexpr auto num_constraints = 5U;
    constexpr auto device_data_host_data = std::array{
        Constraints::DeviceData{
            ConstraintType::kFixedBC, {0U, 6U}, {0U, 0U}, {0U, 0U}, 0U, 0U, {}, {}, {}, {}},
        Constraints::DeviceData{
            ConstraintType::kPrescribedBC, {6U, 12U}, {0U, 0U}, {0U, 0U}, 0U, 0U, {}, {}, {}, {}},
        Constraints::DeviceData{
            ConstraintType::kRigidJoint, {12U, 18U}, {0U, 0U}, {0U, 0U}, 0U, 0U, {}, {}, {}, {}},
        Constraints::DeviceData{
            ConstraintType::kRevoluteJoint, {18U, 24U}, {0U, 0U}, {0U, 0U}, 0U, 0U, {}, {}, {}, {}},
        Constraints::DeviceData{
            ConstraintType::kRotationControl,
            {24U, 30U},
            {0U, 0U},
            {0U, 0U},
            0U,
            0U,
            {},
            {},
            {},
            {}}};
    const auto device_data_host =
        Kokkos::View<const Constraints::DeviceData[num_constraints], Kokkos::HostSpace>(
            device_data_host_data.data()
        );
    const auto device_data = Kokkos::View<Constraints::DeviceData[num_constraints]>("device_data");
    Kokkos::deep_copy(device_data, device_data_host);

    auto nnz = size_t{0U};
    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros_Constraints", num_constraints,
        ComputeNumberOfNonZeros_Constraints{device_data}, nnz
    );

    EXPECT_EQ(nnz, 288U);
}

}  // namespace openturbine::tests
