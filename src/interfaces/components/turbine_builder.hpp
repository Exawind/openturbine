#pragma once

#include "interfaces/components/turbine.hpp"
#include "interfaces/components/turbine_input.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Builder class for creating Turbine objects with a fluent interface pattern
 */
class TurbineBuilder {
public:
    /**
     * @brief Get the current blade input configuration
     * @return Reference to the current blade input
     */
    [[nodiscard]] const TurbineInput& Input() const { return this->input; }

    /**
     * @brief Build a Blade object from the current configuration
     * @param model The model to associate with this blade
     * @return A new Blade object
     */
    [[nodiscard]] Turbine Build(Model& model) const { return {this->input, model}; }

private:
    TurbineInput input;
};

}  // namespace openturbine::interfaces::components
