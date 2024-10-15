#pragma once

namespace openturbine {
struct Solver;
struct Constraints;

void UpdateConstraintPrediction(const Solver& solver, const Constraints& constraints);
}  // namespace openturbine
