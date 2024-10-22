#pragma once

namespace openturbine {

struct StepParameters;
struct Solver;
struct Beams;
struct State;
struct Constraints;

bool Step(
    const StepParameters& parameters, Solver& solver, const Beams& beams, const State& state,
    Constraints& constraints
);

}  // namespace openturbine
