#pragma once

#include <cstddef>

#include "node_data.hpp"

namespace openturbine::cfd {

struct MooringLine {
    /// @brief Fairlead node
    NodeData fairlead_node;

    /// @brief Anchor node
    NodeData anchor_node;

    /// @brief Fixed constraint identifier for the anchor node
    size_t fixed_constraint_id{0U};

    /// @brief Rigid constraint identifier between fairlead and platform nodes
    size_t rigid_constraint_id{0U};

    /// @brief Spring element identifier between fairlead and anchor nodes
    size_t spring_element_id{0U};
};

}  // namespace openturbine::cfd
