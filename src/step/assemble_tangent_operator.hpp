#pragma once

namespace openturbine {
struct Solver;
struct State;

void AssembleTangentOperator(const Solver& solver, const State& state);

}  // namespace openturbine
