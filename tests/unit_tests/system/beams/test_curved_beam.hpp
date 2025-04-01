#pragma once

#include <array>
#include <vector>

namespace openturbine::tests {

/// Number of elements
constexpr size_t kNumElems{1};

/// Number of nodes
constexpr size_t kNumNodes{3};

/// Number of quadrature points
constexpr size_t kNumQPs{7};

/// Tolerance for floating point comparisons (unless otherwise stated)
constexpr double kDefaultTolerance{1e-12};

/// Second order polynomial GLL nodes
const std::vector<double> kGLLNodes{-1., 0., 1.};

/// Node positions for curved beam test (from Mathematica script)
constexpr std::array<std::array<double, 3>, 3> kCurvedBeamNodes = {{
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

/// 7-point Gauss quadrature weights
constexpr std::array<double, 7> kGaussQuadratureWeights = {
    0.1294849661688697,  // weight 1
    0.2797053914892766,  // weight 2
    0.3818300505051189,  // weight 3
    0.4179591836734694,  // weight 4
    0.3818300505051189,  // weight 5
    0.2797053914892766,  // weight 6
    0.1294849661688697   // weight 7
};

/// Expected Lagrange polynomial interpolation weights at quadrature points (from Mathemtica script)
constexpr std::array<std::array<double, 3>, 7> kExpectedInterpWeights = {{
    {0.924956870807, 0.0991941707284, -0.0241510415356},  // QP 1
    {0.645699842408, 0.450131500784, -0.0958313431915},   // QP 2
    {0.285277719137, 0.835289713103, -0.12056743224},     // QP 3
    {0., 1., 0.},                                         // QP 4
    {-0.12056743224, 0.835289713103, 0.285277719137},     // QP 5
    {-0.0958313431915, 0.450131500784, 0.645699842408},   // QP 6
    {-0.0241510415356, 0.0991941707284, 0.924956870807}   // QP 7
}};

/// Expected Lagrange polynomial derivative weights at quadrature points (from Mathemtica script)
constexpr std::array<std::array<double, 3>, 7> kExpectedDerivWeights = {{
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

/// Material properties for curved beam (from Mathemtica script)
constexpr std::array<std::array<double, 6>, 6> kCurvedBeamCuu = {
    std::array<double, 6>{1.252405841673e6, -316160.8714634, 190249.1405163, 0., 0., 0.},
    std::array<double, 6>{-316160.8714634, 174443.0134269, -51312.45787663, 0., 0., 0.},
    std::array<double, 6>{190249.1405163, -51312.45787663, 68661.14489973, 0., 0., 0.},
    std::array<double, 6>{0.0, 0.0, 0.0, 31406.91734868, 25085.26404619, -17572.20634725},
    std::array<double, 6>{0.0, 0.0, 0.0, 25085.26404619, 47547.80267146, 6533.599954035},
    std::array<double, 6>{0.0, 0.0, 0.0, -17572.20634725, 6533.599954035, 138595.2799799}
};

/// Expected strain for curved beam (from Mathematica script)
constexpr std::array<double, 6> kCurvedBeamStrain = {
    0.002691499530001,  // ex
    -0.04310062503412,  // ey
    0.01251033519481,   // ez
    0.09666498438782,   // gxy
    0.09665741323766,   // gxz
    0.001532568414933   // gyz
};

/// Expected Fc i.e. elastic forces for curved beam (from Mathematica script)
constexpr std::array<double, 6> kExpectedFc = {
    19377.66142402,  // Fx
    -9011.48579619,  // Fy
    3582.628416357,  // Fz
    5433.695299839,  // Mx
    7030.727457672,  // My
    -854.6894329742  // Mz
};

/// Strain interpolation holder (einterphold from Mathematica)
constexpr std::array<double, 6> kStrainInterpolationHolder = {
    0.9549992533551,   // ex
    -0.3059615881351,  // ey
    0.167475028213,    // ez
    0.09666498438782,  // gxy
    0.09665741323766,  // gxz
    0.001532568414933  // gyz
};

/// Expected Fd i.e. damping forces for curved beam (from Mathematica script)
constexpr std::array<double, 6> kExpectedFd = {
    0.,               // Fx
    0.,               // Fy
    0.,               // Fz
    -413.0521579912,  // Mx
    176.1330689806,   // My
    2677.142143344    // Mz
};

}  // namespace openturbine::tests
