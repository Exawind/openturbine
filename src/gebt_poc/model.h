#pragma once

#include "src/gebt_poc/section.h"

namespace openturbine::gebt_poc {

/// Class to manages different aspects of the beam finite element model
class Model {
public:
    Model(std::string name) : name_(name) {}

    /// Returns the name of the model
    inline std::string GetName() const { return name_; }

    /// Adds a section to the model
    void AddSection(Section section) { sections_.emplace_back(section); }

    /// Returns the sections of the model
    inline const std::vector<Section>& GetSections() const { return sections_; }

private:
    std::string name_;               //< Name of the model
    std::vector<Section> sections_;  //< Sections of the beam
};

}  // namespace openturbine::gebt_poc
