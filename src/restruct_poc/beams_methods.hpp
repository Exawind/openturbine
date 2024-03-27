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

void CalculateQPData(Beams& beams);

void CalculateResidualVector(Beams& beams, View_N residual_vector);

void CalculateMassMatrix(Beams& beams, View_NxN M);

void CalculateGyroscopicInertiaMatrix(Beams& beams, View_NxN G);

void CalculateInertialStiffnessMatrix(Beams& beams, View_NxN K);

void CalculateElasticStiffnessMatrix(Beams& beams, View_NxN K);

}  // namespace oturb