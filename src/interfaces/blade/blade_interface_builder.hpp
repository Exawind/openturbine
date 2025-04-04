#pragma once

#include "blade_interface.hpp"
#include "interfaces/components/blade_builder.hpp"
#include "interfaces/components/solution_builder.hpp"

namespace openturbine::interfaces {

struct BladeInterfaceBuilder {
    [[nodiscard]] components::BladeBuilder& Blade() { return this->blade_builder; }

    [[nodiscard]] components::SolutionBuilder& Solution() { return this->solution_builder; }

    [[nodiscard]] BladeInterface Build() const {
        return BladeInterface(this->solution_builder.Input(), this->blade_builder.Input());
    }

private:
    components::BladeBuilder blade_builder;
    components::SolutionBuilder solution_builder;
};

}  // namespace openturbine::interfaces
