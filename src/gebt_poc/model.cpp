#include "src/gebt_poc/model.h"

namespace openturbine::gebt_poc {

Model::Model(std::string name) : name_(name) {
}

Model::Model(std::string name, std::vector<Section> sections)
    : name_(name), sections_(std::move(sections)) {
}

}  // namespace openturbine::gebt_poc
