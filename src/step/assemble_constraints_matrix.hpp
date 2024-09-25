#pragma once

namespace openturbine {
struct Solver;
struct Constraints;

void AssembleConstraintsMatrix(Solver& solver, const Constraints& constraints);
}  // namespace openturbine
