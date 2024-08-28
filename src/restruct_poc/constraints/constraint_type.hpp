#pragma once

namespace openturbine {

/// @brief Enum class to define the type of constraint
enum class ConstraintType : std::uint8_t {
    kNone = 0,          // No constraint (default)
    kFixedBC = 1,       // Fixed boundary condition constraint (zero displacement)
    kPrescribedBC = 2,  // Prescribed boundary condition (displacement can be set)
    kRigid = 3,         // nodes maintain relative distance and rotation
    kCylindrical = 4,      // Target node rotates freely around specified axis. Relative distance and
                           // rotation are fixed)
    kRotationControl = 5,  // Specify rotation about given axis
};

KOKKOS_INLINE_FUNCTION
constexpr size_t GetNumberOfNodes(ConstraintType t) {
    const auto has_two_nodes = t == ConstraintType::kRigid || t == ConstraintType::kCylindrical ||
                               t == ConstraintType::kRotationControl;
    return 1U + static_cast<size_t>(has_two_nodes);
}


}
