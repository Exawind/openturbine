#pragma once

namespace openturbine {

struct StepParameters;
struct Solver;
struct State;
struct Constraints;

void PostStepCleanUp(
    const StepParameters& parameters, const Solver& solver, const State& state,
    const Constraints& constraints
);
}  // namespace openturbine
