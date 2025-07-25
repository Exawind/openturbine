#include "node_data.hpp"

#include "host_state.hpp"

namespace openturbine::interfaces {
void NodeData::ClearLoads() {
    this->loads = {0., 0., 0., 0., 0., 0.};
}

void NodeData::GetMotion(const HostState<DeviceType>& host_state) {
    for (auto component = 0U; component < 7U; ++component) {
        this->position[component] = host_state.x(this->id, component);
        this->displacement[component] = host_state.q(this->id, component);
    }
    for (auto component = 0U; component < 6U; ++component) {
        this->velocity[component] = host_state.v(this->id, component);
        this->acceleration[component] = host_state.vd(this->id, component);
    }
}

void NodeData::SetLoads(HostState<DeviceType>& host_state) const {
    for (auto component = 0U; component < 6U; ++component) {
        host_state.f(this->id, component) = this->loads[component];
    }
}
}  // namespace openturbine::interfaces
