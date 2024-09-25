#pragma once

namespace openturbine {
struct StepParameters;
struct Solver;
struct State;

void UpdateStatePrediction(
    const StepParameters& parameters, const Solver& solver, const State& state
);

}  // namespace openturbine
