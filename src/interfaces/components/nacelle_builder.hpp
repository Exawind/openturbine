#pragma once

#include "nacelle_hpp"
#include "nacelle_input.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Builder class for creating Nacelle objects with a fluent interface pattern
 */
class NacelleBuilder {
public:
       /**
     * @brief Get the current blade input configuration
     * @return Reference to the current blade input
     */
    [[nodiscard]] const NacelleInput& Input() const { return this->input; }

    /**
     * @brief Build a Blade object from the current configuration
     * @param model The model to associate with this blade
     * @return A new Blade object
     */
    [[nodiscard]] Nacelle Build(Model& model) const { return {this->input, model}; }

private:
    NacelleInput input;
};

}  // namespace openturbine::interfaces::components
