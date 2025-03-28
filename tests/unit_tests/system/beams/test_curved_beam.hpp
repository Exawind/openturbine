#pragma once

#include <array>
#include <vector>

namespace openturbine::tests {

constexpr double kTolerance = 1e-12;

/// Second order polynomial GLL nodes
const std::vector<double> kGLLNodes = {-1., 0., 1.};

/// Node positions for curved beam test (from Mathematica script)
const std::array<std::array<double, 3>, 3> kCurvedBeamNodes = {{
    {0., 0., 0.},       // Node 1
    {2.5, -0.125, 0.},  // Node 2
    {5., 1., -1.}       // Node 3
}};

/// 7-point Gauss quadrature locations
constexpr std::array<double, 7> kGaussQuadraturePoints = {
    -0.9491079123427585,  // point 1
    -0.7415311855993945,  // point 2
    -0.4058451513773972,  // point 3
    0.,                   // point 4
    0.4058451513773972,   // point 5
    0.7415311855993945,   // point 6
    0.9491079123427585    // point 7
};

/// Expected Lagrange polynomial interpolation weights at quadrature points (from Mathemtica script)
const std::array<std::array<double, 3>, 7> kExpectedInterpWeights = {{
    {0.924956870807, 0.0991941707284, -0.0241510415356},  // QP 1
    {0.645699842408, 0.450131500784, -0.0958313431915},   // QP 2
    {0.285277719137, 0.835289713103, -0.12056743224},     // QP 3
    {0., 1., 0.},                                         // QP 4
    {-0.12056743224, 0.835289713103, 0.285277719137},     // QP 5
    {-0.0958313431915, 0.450131500784, 0.645699842408},   // QP 6
    {-0.0241510415356, 0.0991941707284, 0.924956870807}   // QP 7
}};

/// Expected Lagrange polynomial derivative weights at quadrature points (from Mathemtica script)
const std::array<std::array<double, 3>, 7> kExpectedDerivWeights = {{
    {-1.449107912343, 1.898215824686, -0.4491079123428},    // QP 1
    {-1.241531185599, 1.483062371199, -0.2415311855994},    // QP 2
    {-0.9058451513774, 0.8116903027548, 0.0941548486226},   // QP 3
    {-0.5, 0.0, 0.5},                                       // QP 4
    {-0.0941548486226, -0.8116903027548, 0.9058451513774},  // QP 5
    {0.2415311855994, -1.483062371199, 1.241531185599},     // QP 6
    {0.4491079123428, -1.898215824686, 1.449107912343}      // QP 7
}};

/// Expected Jacobians at quadrature points for curved beam (from Mathemtica script)
constexpr std::array<double, 7> kExpectedJacobians = {
    2.631125640242,  // QP 1
    2.54766419719,   // QP 2
    2.501783068048,  // QP 3
    2.598076211353,  // QP 4
    2.843452426325,  // QP 5
    3.134881687854,  // QP 6
    3.34571483248    // QP 7
};

}  // namespace openturbine::tests
