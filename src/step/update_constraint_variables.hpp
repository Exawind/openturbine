#pragma once

namespace openturbine {
struct State;
struct Constraints;

void UpdateConstraintVariables(const State& state, Constraints& constraints);
}  // namespace openturbine
