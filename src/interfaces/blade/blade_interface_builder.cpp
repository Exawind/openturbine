#include "blade_interface.hpp"
#include "blade_interface_builder.hpp"

namespace openturbine::interfaces {

components::SolutionBuilder& BladeInterfaceBuilder::Solution() { return this->solution_builder; }

components::BeamBuilder& BladeInterfaceBuilder::Blade() { return this->beam_builder; }

BladeInterface BladeInterfaceBuilder::Build() const {
        return BladeInterface(this->solution_builder.Input(), this->beam_builder.Input());
}

}
