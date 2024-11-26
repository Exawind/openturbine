#pragma once

#include <Kokkos_Core.hpp>

#include "src/dof_management/freedom_signature.hpp"
#include "src/types.hpp"

namespace openturbine {

/**
 * @brief Enumeration to define the type of the element
 */
enum class ElementsType : std::uint8_t {
    kBeams = 0,          //< Beam elements
    kMasses = 1,         //< Mass/rigid body elements
    kLinearSprings = 2,  //< Linear spring elements
};

/**
 * @brief Abstract class to define a common interface across all element types
 */
class Elements {
public:
    virtual ~Elements() = default;

    /// Returns the element type
    virtual ElementsType ElementType() const = 0;
};

}  // namespace openturbine
