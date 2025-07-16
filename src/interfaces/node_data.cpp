#include "node_data.hpp"

#include "host_state.hpp"

namespace openturbine::interfaces {
void NodeData::ClearLoads() {
    this->loads = {0., 0., 0., 0., 0., 0.};
}

void NodeData::GetMotion(const HostState<DeviceType>& host_state) {
    for (auto i = 0U; i < 7U; ++i) {
        this->position[i] = host_state.x(this->id, i);
        this->displacement[i] = host_state.q(this->id, i);
    }
    for (auto i = 0U; i < 6U; ++i) {
        this->velocity[i] = host_state.v(this->id, i);
        this->acceleration[i] = host_state.vd(this->id, i);
    }
}

void NodeData::SetLoads(HostState<DeviceType>& host_state) const {
    for (auto i = 0U; i < 6U; ++i) {
        host_state.f(this->id, i) = this->loads[i];
    }
}
}  // namespace openturbine::interfaces
