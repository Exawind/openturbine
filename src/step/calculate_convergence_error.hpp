#pragma once

namespace openturbine {
struct Solver;
struct State;

double CalculateConvergenceError(const Solver& solver, const State& state);

}  // namespace openturbine
