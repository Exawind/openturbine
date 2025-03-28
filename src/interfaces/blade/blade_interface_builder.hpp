#pragma once

#include "blade_interface.hpp"
#include "interfaces/components/blade_builder.hpp"
#include "interfaces/components/solution_builder.hpp"

namespace openturbine::interfaces {

using namespace openturbine::interfaces::components;

struct BladeInterfaceBuilder {
    [[nodiscard]] BladeBuilder& Blade() { return this->blade_builder; }

    [[nodiscard]] SolutionBuilder& Solution() { return this->solution_builder; }

    [[nodiscard]] BladeInterface Build() const {
        return BladeInterface(this->solution_builder.Input(), this->blade_builder.Input());
    }

private:
    BladeBuilder blade_builder;
    SolutionBuilder solution_builder;
};

}  // namespace openturbine::interfaces
