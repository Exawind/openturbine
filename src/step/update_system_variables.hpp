#pragma once

namespace openturbine {
struct StepParameters;
struct Beams;
struct State;

void UpdateSystemVariables(const StepParameters& parameters, const Beams& beams, const State& state);

}  // namespace openturbine
