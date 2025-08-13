#pragma once

#include <array>
#include <cstddef>

namespace openturbine::interfaces::cfd {

struct NodeData {
    size_t id;
    std::array<double, 7> position{0., 0., 0., 1., 0., 0., 0.};
    std::array<double, 7> displacement{0., 0., 0., 1., 0., 0., 0.};
    std::array<double, 6> velocity{0., 0., 0., 0., 0., 0.};
    std::array<double, 6> acceleration{0., 0., 0., 0., 0., 0.};
    std::array<double, 6> loads{0., 0., 0., 0., 0., 0.};

    explicit NodeData(size_t id_) : id(id_) {}
};

}  // namespace openturbine::cfd
