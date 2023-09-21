#pragma once

#include "src/gebt_poc/section.h"

namespace openturbine::gebt_poc {

/// Creates a Kokkos view of sections from a provided std::vector of sections
Kokkos::View<Section*> CreateSections(std::vector<Section>);

/// Class to manage different aspects of the beam finite element model creation
class Model {
public:
    Model(std::string name, Kokkos::View<Section*>);

    /// Returns the name of the model
    inline std::string GetName() const { return name_; }

    /// Returns the sections of the model
    inline Kokkos::View<Section*> GetSections() const { return sections_; }

private:
    std::string name_;                 //< Name of the model
    Kokkos::View<Section*> sections_;  //< Sections of the beam
};

}  // namespace openturbine::gebt_poc
