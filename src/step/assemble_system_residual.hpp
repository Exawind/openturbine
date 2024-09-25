#pragma once

namespace openturbine {
struct Solver;
struct Beams;

void AssembleSystemResidual(const Solver& solver, const Beams& beams);

}  // namespace openturbine
