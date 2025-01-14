#pragma once

#include "node_data.hpp"

namespace openturbine::cfd {

struct MooringLine {
    /// @brief Fairlead node
    NodeData fairlead_node;

    /// @brief Anchor node
    NodeData anchor_node;

    /// @brief Spring element identifier
    size_t spring_element_id = 0;

    /// @brief Rigid constraint identifier
    size_t rigid_constraint_id = 0;

    MooringLine(
        size_t fairlead_node_id, size_t anchor_node_id, size_t spring_element_id_,
        size_t rigid_constraint_id_
    )
        : fairlead_node(fairlead_node_id),
          anchor_node(anchor_node_id),
          spring_element_id(spring_element_id_),
          rigid_constraint_id(rigid_constraint_id_) {}
};

}