#pragma once

namespace openturbine {
struct Solver;
struct Constraints;

void AssembleConstraintsResidual(const Solver& solver, const Constraints& constraints);

}  // namespace openturbine
