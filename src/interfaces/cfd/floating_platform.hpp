#pragma once

#include "vector"

#include "src/interfaces/cfd/mooring_line.hpp"
#include "src/interfaces/cfd/node_data.hpp"

namespace openturbine::cfd {

//------------------------------------------------------------------------------
// Data Structures
//------------------------------------------------------------------------------

struct FloatingPlatform {
    /// @brief Flag indicating platform is active in model
    bool active = false;

    /// @brief Platform node data (ID, motion, loads)
    NodeData node{0};

    /// @brief Platform mass element identifier
    size_t mass_element_id = 0;

    /// @brief Mooring line data array
    std::vector<MooringLine> mooring_lines;
};

}  // namespace openturbine::cfd
