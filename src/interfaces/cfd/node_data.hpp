#pragma once

#include "types.hpp"

namespace openturbine::cfd {

struct NodeData {
    size_t id;
    Array_7 position{0., 0., 0., 1., 0., 0., 0.};
    Array_7 displacement{0., 0., 0., 1., 0., 0., 0.};
    Array_6 velocity{0., 0., 0., 0., 0., 0.};
    Array_6 acceleration{0., 0., 0., 0., 0., 0.};
    Array_6 loads{0., 0., 0., 0., 0., 0.};

    explicit NodeData(size_t id_) : id(id_) {}
};

}  // namespace openturbine::cfd
