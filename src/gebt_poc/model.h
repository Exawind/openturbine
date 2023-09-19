#pragma once

#include "src/gebt_poc/section.h"

namespace openturbine::gebt_poc {

/// Class to manages different aspects of the beam finite element model
class Model {
public:
    Model(std::string name, Kokkos::View<Section*> sections);

    /// Returns the name of the model
    inline std::string GetName() const { return name_; }

    /// Returns the sections of the model
    inline Kokkos::View<Section*> GetSections() const { return sections_; }

private:
    std::string name_;                 //< Name of the model
    Kokkos::View<Section*> sections_;  //< Sections of the beam
};

}  // namespace openturbine::gebt_poc
