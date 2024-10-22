#pragma once

namespace openturbine {
struct StepParameters;
struct State;

void PredictNextState(const StepParameters& parameters, const State& state);

}  // namespace openturbine
