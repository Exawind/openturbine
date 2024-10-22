#pragma once

namespace openturbine {
struct Solver;
struct Beams;

void AssembleSystemMatrix(Solver& solver, const Beams& beams);
}  // namespace openturbine
