#pragma once

namespace openturbine {
struct Beams;

void AssembleInertiaMatrix(const Beams& beams, double beta_prime, double gamma_prime);

}  // namespace openturbine
