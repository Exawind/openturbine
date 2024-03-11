#pragma once

#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

/// @brief Class to store and manage the states of a dynamic system
struct State {
    LieGroupFieldView generalized_coordinates;
    LieAlgebraFieldView velocity;
    LieAlgebraFieldView acceleration;
    LieAlgebraFieldView algorithmic_acceleration;
};
}  // namespace openturbine::gebt_poc
