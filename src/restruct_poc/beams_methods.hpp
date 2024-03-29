#pragma once

#include "beams_data.hpp"
#include "beams_input.hpp"

namespace oturb {

void LagrangePolynomialInterpWeights(
    const double x, const std::vector<double>& xs, std::vector<double>& weights
);

void LagrangePolynomialDerivWeights(
    const double x, const std::vector<double>& xs, std::vector<double>& weights
);

void UpdateState(Beams& beams, View_Nx7 Q, View_Nx6 V, View_Nx6 A);

void AssembleResidualVector(Beams& beams, View_N residual_vector);

void AssembleMassMatrix(Beams& beams, View_NxN M);

void AssembleGyroscopicInertiaMatrix(Beams& beams, View_NxN G);

void AssembleInertialStiffnessMatrix(Beams& beams, View_NxN K);

void AssembleElasticStiffnessMatrix(Beams& beams, View_NxN K);

}  // namespace oturb