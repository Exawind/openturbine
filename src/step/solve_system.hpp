#pragma once

namespace openturbine {
struct StepParameters;
struct Solver;

void SolveSystem(const StepParameters& parameters, Solver& solver);
}  // namespace openturbine
