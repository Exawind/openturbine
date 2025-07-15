#pragma once

namespace openturbine::interfaces {

struct ConstraintData {
    /// @brief Node identifier in model
    size_t id;

    /// @brief Constraint data constructor
    /// @param id Constraint identifier in model
    explicit ConstraintData(size_t id_) : id(id_) {}
};

}  // namespace openturbine::interfaces
