#pragma once

namespace openturbine {
struct StepParameters;
struct State;

void UpdateTangentOperator(const StepParameters& parameters, const State& state);

}  // namespace openturbine
