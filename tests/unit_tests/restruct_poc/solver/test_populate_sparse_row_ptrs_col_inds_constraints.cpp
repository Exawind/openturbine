#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/solver/populate_sparse_row_ptrs_col_inds_constraints.hpp"

namespace openturbine::tests {

TEST(PopulateSparseRowPtrsColInds_Constraints, OneOfEach) {
    constexpr auto num_constraints = 5U;
    constexpr auto num_constraint_dofs = 29U;
    constexpr auto num_non_zero = 288U;
    constexpr auto device_data_host_data = 
        std::array{Constraints::DeviceData{ConstraintType::kFixedBC, {0U, 6U}, {0U, 0U}, {0U, 0U}, 1U, 2U, {}, {}, {}, {}}, 
                   Constraints::DeviceData{ConstraintType::kPrescribedBC, {6U, 12U}, {0U, 0U}, {0U, 0U}, 3U, 4U, {}, {}, {}, {}},
                   Constraints::DeviceData{ConstraintType::kRigid, {12U, 18U}, {0U, 0U}, {0U, 0U}, 5U, 6U, {}, {}, {}, {}},
                   Constraints::DeviceData{ConstraintType::kCylindrical, {18U, 23U}, {0U, 0U}, {0U, 0U}, 7U, 8U, {}, {}, {}, {}},
                   Constraints::DeviceData{ConstraintType::kRotationControl, {23U, 29U}, {0U, 0U}, {0U, 0U}, 9U, 10U, {}, {}, {}, {}}};
    const auto device_data_host = Kokkos::View<const Constraints::DeviceData[num_constraints], Kokkos::HostSpace>(device_data_host_data.data());
    const auto device_data = Kokkos::View<Constraints::DeviceData[num_constraints]>("device_data");
    Kokkos::deep_copy(device_data, device_data_host);

    const auto B_row_ptrs = Kokkos::View<size_t[num_constraint_dofs + 1U]>("B_row_ptrs");
    const auto B_col_inds = Kokkos::View<size_t[num_non_zero]>("B_col_inds");

    Kokkos::parallel_for("PopulateSparseRowPtrsColInds_Constraints", 1, PopulateSparseRowPtrsColInds_Constraints<Kokkos::View<size_t*>, Kokkos::View<size_t*>>{device_data, B_row_ptrs, B_col_inds});

    constexpr auto B_row_ptrs_exact_data = std::array<size_t, num_constraint_dofs + 1U>{0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276};
    const auto B_row_ptrs_exact = Kokkos::View<const size_t[num_constraint_dofs + 1U], Kokkos::HostSpace>(B_row_ptrs_exact_data.data());

    const auto B_row_ptrs_host = Kokkos::create_mirror(B_row_ptrs);
    Kokkos::deep_copy(B_row_ptrs_host, B_row_ptrs);

    for(auto i = 0U; i < num_constraint_dofs + 1U; ++i) {
        EXPECT_EQ(B_row_ptrs_host(i), B_row_ptrs_exact(i));
    }

    constexpr auto B_col_inds_exact_data = std::array<size_t, num_non_zero>{12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29, 24, 25, 26, 27, 28, 29, 24, 25, 26, 27, 28, 29, 24, 25, 26, 27, 28, 29, 24, 25, 26, 27, 28, 29, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 30, 31, 32, 33, 34, 35, 48, 49, 50, 51, 52, 53, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 42, 43, 44, 45, 46, 47, 60, 61, 62, 63, 64, 65, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 54, 55, 56, 57, 58, 59, };
    const auto B_col_inds_exact = Kokkos::View<const size_t[num_non_zero], Kokkos::HostSpace>(B_col_inds_exact_data.data());

    const auto B_col_inds_host = Kokkos::create_mirror(B_col_inds);
    Kokkos::deep_copy(B_col_inds_host, B_col_inds);

    for(auto i = 0U; i < num_non_zero; ++i) {
        EXPECT_EQ(B_col_inds_host(i), B_col_inds_exact(i));
    }
}

}
