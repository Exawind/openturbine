#pragma once

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

namespace openturbine::tests::curved_beam {

//--------------------------------------------------------------------------
// FEA inputs/settings for curved beam
//--------------------------------------------------------------------------

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
constexpr std::array<double, 21> kCurvedBeamNodes_data = {
    0.,  0.,     0.,  0., 0., 0., 0.,  // Node 1
    2.5, -0.125, 0.,  0., 0., 0., 0.,  // Node 2
    5.,  1.,     -1., 0., 0., 0., 0.   // Node 3
};
const Kokkos::View<double[1][3][7], Kokkos::HostSpace>::const_type kCurvedBeamNodes(
    kCurvedBeamNodes_data.data()
);

/// 7-point Gauss quadrature locations (high order quadrature rule)
constexpr std::array<double, 7> kGaussQuadraturePoints = {
    -0.9491079123427585,  // point 1
    -0.7415311855993945,  // point 2
    -0.4058451513773972,  // point 3
    0.,                   // point 4
    0.4058451513773972,   // point 5
    0.7415311855993945,   // point 6
    0.9491079123427585    // point 7
};

/// 7-point Gauss quadrature weights (high order quadrature rule)
constexpr std::array<double, 7> kGaussQuadratureWeights = {
    0.1294849661688697,  // weight 1
    0.2797053914892766,  // weight 2
    0.3818300505051189,  // weight 3
    0.4179591836734694,  // weight 4
    0.3818300505051189,  // weight 5
    0.2797053914892766,  // weight 6
    0.1294849661688697   // weight 7
};

//--------------------------------------------------------------------------
// Shape function and jacobian calculations: inputs and expected values
//--------------------------------------------------------------------------

/// Expected Lagrange polynomial interpolation weights at quadrature points (from Mathemtica script)
constexpr std::array<std::array<double, 3>, 7> kExpectedInterpWeights = {{
    {0.924956870807194, 0.0991941707283707, -0.02415104153556458},  // QP 1
    {0.6456998424079191, 0.4501315007835565, -0.0958313431914754},  // QP 2
    {0.2852777191369698, 0.835289713103458, -0.1205674322404274},   // QP 3
    {0., 1., 0.},                                                   // QP 4
    {-0.1205674322404274, 0.835289713103458, 0.2852777191369698},   // QP 5
    {-0.0958313431914754, 0.4501315007835565, 0.6456998424079191},  // QP 6
    {-0.02415104153556458, 0.0991941707283707, 0.924956870807194}   // QP 7
}};

/// Expected Lagrange polynomial derivative weights at quadrature points (from Mathemtica script)
constexpr std::array<std::array<double, 3>, 7> kExpectedDerivWeights = {{
    {-1.449107912342756, 1.898215824685517, -0.4491079123427585},  // QP 1
    {-1.241531185599395, 1.483062371198789, -0.2415311855993944},  // QP 2
    {-0.905845151377397, 0.811690302754794, 0.0941548486226028},   // QP 3
    {-0.5, 0.0, 0.5},                                              // QP 4
    {-0.0941548486226028, -0.811690302754794, 0.905845151377397},  // QP 5
    {0.2415311855993944, -1.483062371198789, 1.241531185599395},   // QP 6
    {0.4491079123427585, -1.898215824685517, 1.449107912342756}    // QP 7
}};

/// Expected Jacobians at quadrature points for curved beam (from Mathemtica script)
constexpr std::array<double, 7> kExpectedJacobians = {
    2.631125640241708,  // QP 1
    2.547664197189947,  // QP 2
    2.501783068048316,  // QP 3
    2.598076211353316,  // QP 4
    2.843452426324649,  // QP 5
    3.134881687853751,  // QP 6
    3.345714832480474   // QP 7
};

//--------------------------------------------------------------------------
// Forces calculations: inputs and expected values
//--------------------------------------------------------------------------

/// Material properties for curved beam (from Mathemtica script)
constexpr std::array<double, 36> kCurvedBeamCuu_data = {
    1.252405841673e6,
    -316160.8714634,
    190249.1405163,
    0.,
    0.,
    0.,
    -316160.8714634,
    174443.0134269,
    -51312.45787663,
    0.,
    0.,
    0.,
    190249.1405163,
    -51312.45787663,
    68661.14489973,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    31406.91734868,
    25085.26404619,
    -17572.20634725,
    0.,
    0.,
    0.,
    25085.26404619,
    47547.80267146,
    6533.599954035,
    0.,
    0.,
    0.,
    -17572.20634725,
    6533.599954035,
    138595.2799799
};
const Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type kCurvedBeamCuu(
    kCurvedBeamCuu_data.data()
);

/// Expected strain for curved beam (from Mathematica script)
constexpr std::array<double, 6> kCurvedBeamStrain_data = {
    0.002691499530001,  // ex
    -0.04310062503412,  // ey
    0.01251033519481,   // ez
    0.09666498438782,   // gxy
    0.09665741323766,   // gxz
    0.001532568414933   // gyz
};
const Kokkos::View<double[6], Kokkos::HostSpace>::const_type kCurvedBeamStrain(
    kCurvedBeamStrain_data.data()
);

/// Expected Fc i.e. elastic forces for curved beam (from Mathematica script)
constexpr std::array<double, 6> kExpectedFc_data = {
    19377.66142402,  // Fx
    -9011.48579619,  // Fy
    3582.628416357,  // Fz
    5433.695299839,  // Mx
    7030.727457672,  // My
    -854.6894329742  // Mz
};
const Kokkos::View<double[6], Kokkos::HostSpace>::const_type kExpectedFc(kExpectedFc_data.data());

/// Strain interpolation holder (einterphold from Mathematica)
constexpr std::array<double, 6> kStrainInterpolationHolder = {
    0.9549992533551,   // ex
    -0.3059615881351,  // ey
    0.167475028213,    // ez
    0.09666498438782,  // gxy
    0.09665741323766,  // gxz
    0.001532568414933  // gyz
};

constexpr std::array<double, 9> kX0pupSS_data = {
    0.,  //
    -kStrainInterpolationHolder[2],
    kStrainInterpolationHolder[1],
    kStrainInterpolationHolder[2],
    0.,
    -kStrainInterpolationHolder[0],
    -kStrainInterpolationHolder[1],
    kStrainInterpolationHolder[0],
    0.
};
const Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type kX0pupSS(kX0pupSS_data.data());

/// Expected Fd i.e. damping forces for curved beam (from Mathematica script)
constexpr std::array<double, 6> kExpectedFd_data = {
    0.,               // Fx
    0.,               // Fy
    0.,               // Fz
    -413.0521579912,  // Mx
    176.1330689806,   // My
    2677.142143344    // Mz
};
const Kokkos::View<double[6], Kokkos::HostSpace>::const_type kExpectedFd(kExpectedFd_data.data());

//--------------------------------------------------------------------------
// Residual calculation: inputs and expected values
//--------------------------------------------------------------------------

constexpr std::array<double, 21> kInterpWeightsFlat = {
    0.924956870807194,
    0.6456998424079191,
    0.2852777191369698,
    0.,
    -0.1205674322404274,
    -0.0958313431914754,
    -0.02415104153556458,
    0.0991941707283707,
    0.4501315007835565,
    0.835289713103458,
    1.,
    0.835289713103458,
    0.4501315007835565,
    0.0991941707283707,
    -0.02415104153556458,
    -0.0958313431914754,
    -0.1205674322404274,
    0.,
    0.2852777191369698,
    0.6456998424079191,
    0.924956870807194
};

constexpr std::array<double, 21> kDerivWeightsFlat = {
    -1.449107912342756,
    -1.241531185599395,
    -0.905845151377397,
    -0.5,
    -0.0941548486226028,
    0.2415311855993944,
    0.4491079123427585,
    1.898215824685517,
    1.483062371198789,
    0.811690302754794,
    0.,
    -0.811690302754794,
    -1.483062371198789,
    -1.898215824685517,
    -0.4491079123427585,
    -0.2415311855993944,
    0.0941548486226028,
    0.5,
    0.905845151377397,
    1.241531185599395,
    1.449107912342756
};

constexpr std::array<double, kNumQPs * 6> kFc = {
    19377.66142402,  -9011.48579619,  3582.628416357,
    5433.695299839,  7030.727457672,  -854.6894329742,  // QP 1
    35490.84254734,  -8099.471990103, 3825.137533463,
    4593.228131633,  7488.269611072,  462.8805414784,  // QP 2
    41809.98986269,  -1118.980021592, 245.0189635062,
    3816.007424285,  7192.596693563,  3198.706243446,  // QP 3
    31320.89196032,  2121.925781777,  561.9839478968,
    4142.167385399,  5587.314015927,  6168.105064308,  // QP 4
    12445.64394842,  -11596.67035144, 8558.832424273,
    5381.871405552,  3716.839224054,  7767.833231818,  // QP 4
    -5219.017496402, -45587.00752485, 21423.97321937,
    6526.890109692,  2581.983143471,  8106.892961828,  // QP 4
    -13939.23015166, -81527.20084932, 31279.33745168,
    7133.754559318,  2143.593577595,  8032.504691472  // QP 4
};

constexpr std::array<double, kNumQPs * 6> kFd = {
    0., 0., 0., -413.0521579912, 176.1330689806,  2677.142143344,  // QP 1
    0., 0., 0., -175.2982322707, -336.0628883725, 914.8821391247,  // QP 2
    0., 0., 0., -27.38979513032, -976.3905881431, 214.6996893003,  // QP 3
    0., 0., 0., -271.4997840407, 3005.929623871,  3781.702029117,  // QP 4
    0., 0., 0., -1251.778505922, 10657.37400997,  16260.30700145,  // QP 5
    0., 0., 0., -198.0160029009, 18894.3224851,   40156.0515182,   // QP 6
    0., 0., 0., 3609.312381055,  24425.88953965,  65272.65615984   // QP 7
};

constexpr std::array<double, kNumQPs * 6> kFi = {
    0.02197622144767,  -0.03476996186535, 0.005820529971857,
    -0.04149911138042, -0.07557306419557, -0.1562386708521,  // QP 1
    0.1114870315503,   -0.152566078275,   0.05693583601535,
    -0.2030250329967,  -0.2770402248744,  -0.6107864532796,  // QP 2
    0.2628772156508,   -0.2574202455274,  0.2344640226209,
    -0.3972952233143,  -0.4088037485627,  -0.7317466112768,  // QP 3
    0.465768653659,    -0.2352745270237,  0.6117152142429,
    -0.4901303217258,  -0.4360315727221,  -0.009475285034617,  // QP 4
    0.6891847954765,   -0.04853357195211, 1.173756627334,
    -0.5474240295309,  -0.278155001394,   1.43370657115,  // QP 5
    0.8867303188692,   0.2236005514299,   1.78060490608,
    -0.776497831473,   0.03946854517767,  3.073630635099,  // QP 6
    1.017116465926,    0.4412924409434,   2.219355473579,
    -1.091189173302,   0.2678910453445,   4.286793316266  // QP 7
};

/// Expected residual forces (from Mathematica notebook)
constexpr std::array<double, 18> kExpectedResidualVector = {
    -38577.92446488, -2956.897670251, 755.6410891537, -4213.441305763, -11013.30888115,
    -6523.841494125, 34178.85505313,  36621.14643499, -16681.89312743, -3152.187800704,
    24867.91997485,  31292.87867203,  4401.879649647, -33664.53616728, 15930.979968,
    6807.108584957,  27016.10156256,  62253.53739314
};

//--------------------------------------------------------------------------
// Linearization matrices calculations: inputs and expected values
//--------------------------------------------------------------------------

constexpr auto xr_data = std::array{0.8628070705148, 0., 0., 0.5055333412048};
const Kokkos::View<double[4], Kokkos::HostSpace>::const_type kCurvedBeamXr(xr_data.data());

constexpr auto Cstar_data = std::array{
    1.36817e6, 0.,     0.,     0.,     0.,     0.,      // row 1
    0.,        88560., 0.,     0.,     0.,     0.,      // row 2
    0.,        0.,     38780., 0.,     0.,     0.,      // row 3
    0.,        0.,     0.,     16960., 17610., -351.,   // row 4
    0.,        0.,     0.,     17610., 59120., -370.,   // row 5
    0.,        0.,     0.,     -351.,  -370.,  141470.  // row 6
};
const Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type kCurvedBeamCstar(Cstar_data.data());

constexpr auto kExpectedCuu_data = std::array{
    394381.5594951,
    545715.5847999,
    0.,
    0.,
    0.,
    0.,  // row 1
    545715.5847999,
    1.062348440505e6,
    0.,
    0.,
    0.,
    0.,  // row 2
    0.,
    0.,
    38780.,
    0.,
    0.,
    0.,  // row 3
    0.,
    0.,
    0.,
    34023.65045212,
    -27172.54931561,
    151.1774277346,  // row 4
    0.,
    0.,
    0.,
    -27172.54931561,
    42056.34954788,
    -487.0794445915,  // row 5
    0.,
    0.,
    0.,
    151.1774277346,
    -487.0794445915,
    141470.  // row 6
};
const Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type kExpectedCuu(kExpectedCuu_data.data()
);

constexpr std::array<double, 9> kCurvedBeamM_tilde_data = {
    0, 854.6894329742,  7030.727457672,  -854.6894329742,
    0, -5433.695299839, -7030.727457672, 5433.695299839,
    0
};
const Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type kCurvedBeamM_tilde(
    kCurvedBeamM_tilde_data.data()
);

constexpr std::array<double, 9> kCurvedBeamN_tilde_data = {
    0, -3582.628416357, -9011.48579619, 3582.628416357,
    0, -19377.66142402, 9011.48579619,  19377.66142402,
    0
};
const Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type kCurvedBeamN_tilde(
    kCurvedBeamN_tilde_data.data()
);

constexpr std::array<double, 36> kExpectedOuu_data = {
    0.,
    0.,
    0.,
    5259.878305537,
    -24476.28810745,
    -72243.1983242,
    0.,
    0.,
    0.,
    9932.579075823,
    3945.69190817,
    -50482.20381259,
    0.,
    0.,
    0.,
    3402.631809939,
    14331.70051426,
    -9205.570213707,
    0.,
    0.,
    0.,
    0.,
    -854.6894329742,
    -7030.727457672,
    0.,
    0.,
    0.,
    854.6894329742,
    0.,
    5433.695299839,
    0.,
    0.,
    0.,
    7030.727457672,
    -5433.695299839,
    0.
};
const Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type kExpectedOuu(kExpectedOuu_data.data()
);

constexpr std::array<double, 36> kExpectedPuu_data = {
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    5259.878305537,
    9932.579075823,
    3402.631809939,
    0.,
    0.,
    0.,
    -24476.28810745,
    3945.69190817,
    14331.70051426,
    0.,
    0.,
    0.,
    -72243.1983242,
    -50482.20381259,
    -9205.570213707,
    0.,
    0.,
    0.
};
const Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type kExpectedPuu(kExpectedPuu_data.data()
);

constexpr std::array<double, 36> kExpectedQuu_data = {
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    2704.533593359,
    5045.754713661,
    -11271.05939004,
    0.,
    0.,
    0.,
    2368.612570317,
    17785.93033178,
    3307.618996744,
    0.,
    0.,
    0.,
    -11094.92632106,
    3720.671154735,
    70314.11063997
};
const Kokkos::View<double[6][6], Kokkos::HostSpace>::const_type kExpectedQuu(kExpectedQuu_data.data()
);

//--------------------------------------------------------------------------
// Stiffness matrix calculation: inputs and expected values
//--------------------------------------------------------------------------

constexpr std::array<double, kNumQPs * 6 * 6> kKuu = {
    0.,
    0.,
    0.,
    -0.004824577014466,
    0.002476462896877,
    0.0001984742319671,
    0.,
    0.,
    0.,
    -0.000123548368338,
    -0.0004996640650474,
    0.0003233068018418,
    0.,
    0.,
    0.,
    -0.0005235893809351,
    0.007820383393196,
    -0.004246242951947,
    0.,
    0.,
    0.,
    0.06338277430594,
    -0.07861048885059,
    0.02343229796372,
    0.,
    0.,
    0.,
    0.1398373890765,
    0.02039789403638,
    -0.05784606450894,
    0.,
    0.,
    0.,
    0.07661013376996,
    0.0195864445657,
    -0.03644475506229,  // QP 1
    0.,
    0.,
    0.,
    -0.01384933501849,
    0.01223684667611,
    -0.0009672551523009,
    0.,
    0.,
    0.,
    -0.00310201402466,
    0.004770406166994,
    -0.001319460677143,
    0.,
    0.,
    0.,
    -0.01264557599321,
    0.03684323704701,
    -0.01483160212693,
    0.,
    0.,
    0.,
    0.2946992941209,
    -0.1880176829137,
    -0.01874159264415,
    0.,
    0.,
    0.,
    0.4600053010099,
    0.09804368435107,
    -0.241999739321,
    0.,
    0.,
    0.,
    0.5033831597769,
    0.1890666625241,
    -0.3092558324376,  // QP 2
    0.,
    0.,
    0.,
    0.003260976624684,
    0.0230974531951,
    -0.01408388479054,
    0.,
    0.,
    0.,
    -0.01546255512659,
    0.04519621636955,
    -0.01614360921848,
    0.,
    0.,
    0.,
    -0.06079695739996,
    0.0766533716668,
    -0.009312652749008,
    0.,
    0.,
    0.,
    0.5503582357982,
    0.4384649536056,
    -0.5226193614784,
    0.,
    0.,
    0.,
    0.1568805119414,
    0.2699286818286,
    -0.2266600717682,
    0.,
    0.,
    0.,
    1.418819963153,
    0.5528568472243,
    -1.03778033758,  // QP 3
    0.,
    0.,
    0.,
    0.06779906433958,
    0.01245928865497,
    -0.04903383614899,
    0.,
    0.,
    0.,
    -0.0390871649622,
    0.1562221749127,
    -0.06314095047007,
    0.,
    0.,
    0.,
    -0.1588048124012,
    0.1193669069116,
    0.0317515652981,
    0.,
    0.,
    0.,
    0.7820579988357,
    2.502335031077,
    -1.885025361362,
    0.,
    0.,
    0.,
    -1.355439103649,
    0.9052417403426,
    0.3345475955845,
    0.,
    0.,
    0.,
    2.738611850442,
    0.6717751977138,
    -2.074890152212,  // QP 4
    0.,
    0.,
    0.,
    0.179225690196,
    -0.0504278664821,
    -0.08796888421767,
    0.,
    0.,
    0.,
    -0.05997102099521,
    0.3325193628813,
    -0.1588448691854,
    0.,
    0.,
    0.,
    -0.3045415941908,
    0.1641671040959,
    0.1026647966214,
    0.,
    0.,
    0.,
    1.273861817188,
    5.779508547484,
    -4.286475919574,
    0.,
    0.,
    0.,
    -3.945398584414,
    2.543877674178,
    1.081270606942,
    0.,
    0.,
    0.,
    4.337812567413,
    0.3102014818512,
    -3.042175178765,  // QP 5
    0.,
    0.,
    0.,
    0.3134611194091,
    -0.1644264304029,
    -0.09782348525142,
    0.,
    0.,
    0.,
    -0.0519909405681,
    0.5169081383567,
    -0.2864342805541,
    0.,
    0.,
    0.,
    -0.4769952302788,
    0.2096583793003,
    0.173923712019,
    0.,
    0.,
    0.,
    2.295665692034,
    9.279992506243,
    -7.197022470367,
    0.,
    0.,
    0.,
    -6.996986562616,
    5.128516896824,
    1.28220430901,
    0.,
    0.,
    0.,
    5.824072172048,
    0.03984501292409,
    -3.730584349658,  // QP 6
    0.,
    0.,
    0.,
    0.4188187366789,
    -0.2731205677995,
    -0.08112967194346,
    0.,
    0.,
    0.,
    -0.01861602233246,
    0.6433787521597,
    -0.388013832628,
    0.,
    0.,
    0.,
    -0.6179229670203,
    0.2446619647178,
    0.2179066413494,
    0.,
    0.,
    0.,
    3.467458733216,
    11.77178912407,
    -9.377738413639,
    0.,
    0.,
    0.,
    -9.41796221498,
    7.53281683069,
    0.9612831821134,
    0.,
    0.,
    0.,
    6.670718693934,
    0.175879406993,
    -4.100113240636  // QP 7
};

constexpr std::array<double, kNumQPs * 6 * 6> kOuu = {
    0.,
    0.,
    0.,
    5259.878305537,
    -24476.28810745,
    -72243.1983242,
    0.,
    0.,
    0.,
    9932.579075823,
    3945.69190817,
    -50482.20381259,
    0.,
    0.,
    0.,
    3402.631809939,
    14331.70051426,
    -9205.570213707,
    0.,
    0.,
    0.,
    0.,
    -854.6894329742,
    -7030.727457672,
    0.,
    0.,
    0.,
    854.6894329742,
    0.,
    5433.695299839,
    0.,
    0.,
    0.,
    7030.727457672,
    -5433.695299839,
    0.,  // QP 1
    0.,
    0.,
    0.,
    -22454.24793324,
    -151208.9545336,
    -68331.25525314,
    0.,
    0.,
    0.,
    10599.75601088,
    24973.00181021,
    -44427.76893969,
    0.,
    0.,
    0.,
    62.82457748333,
    3339.00202713,
    -2518.753876971,
    0.,
    0.,
    0.,
    0.,
    462.8805414784,
    -7488.269611072,
    0.,
    0.,
    0.,
    -462.8805414784,
    0.,
    4593.228131633,
    0.,
    0.,
    0.,
    7488.269611072,
    -4593.228131633,
    0.,  // QP 2
    0.,
    0.,
    0.,
    -4477.215986944,
    -313742.8073858,
    -84956.030414,
    0.,
    0.,
    0.,
    2320.228181973,
    -5534.800252915,
    -54414.69222592,
    0.,
    0.,
    0.,
    922.5800439328,
    64113.56542481,
    10012.01623986,
    0.,
    0.,
    0.,
    0.,
    3198.706243446,
    -7192.596693563,
    0.,
    0.,
    0.,
    -3198.706243446,
    0.,
    3816.007424285,
    0.,
    0.,
    0.,
    7192.596693563,
    -3816.007424285,
    0.,  // QP 3
    0.,
    0.,
    0.,
    57146.05831657,
    -371914.8053116,
    -147555.5970777,
    0.,
    0.,
    0.,
    8907.36195072,
    -109165.1287529,
    -111388.4238159,
    0.,
    0.,
    0.,
    -30181.83797779,
    167157.4368918,
    52019.0704363,
    0.,
    0.,
    0.,
    0.,
    6168.105064308,
    -5587.314015927,
    0.,
    0.,
    0.,
    -6168.105064308,
    0.,
    4142.167385399,
    0.,
    0.,
    0.,
    5587.314015927,
    -4142.167385399,
    0.,  // QP 4
    0.,
    0.,
    0.,
    77183.993751,
    -298162.4525991,
    -208756.3629627,
    0.,
    0.,
    0.,
    24818.05123915,
    -206798.6434849,
    -248402.3814645,
    0.,
    0.,
    0.,
    -75995.34009323,
    219172.5050552,
    129614.6497339,
    0.,
    0.,
    0.,
    0.,
    7767.833231818,
    -3716.839224054,
    0.,
    0.,
    0.,
    -7767.833231818,
    0.,
    5381.871405552,
    0.,
    0.,
    0.,
    3716.839224054,
    -5381.871405552,
    0.,  // QP 5
    0.,
    0.,
    0.,
    51002.97035598,
    -191403.9721169,
    -202368.6385284,
    0.,
    0.,
    0.,
    6124.942615735,
    -246041.6052807,
    -423499.5315683,
    0.,
    0.,
    0.,
    -108235.4341767,
    216147.6081407,
    195038.6349247,
    0.,
    0.,
    0.,
    0.,
    8106.892961828,
    -2581.983143471,
    0.,
    0.,
    0.,
    -8106.892961828,
    0.,
    6526.890109692,
    0.,
    0.,
    0.,
    2581.983143471,
    -6526.890109692,
    0.,  // QP 6
    0.,
    0.,
    0.,
    26641.25345202,
    -123293.0646233,
    -155696.9812347,
    0.,
    0.,
    0.,
    -30784.92940852,
    -249519.7291353,
    -547946.4435877,
    0.,
    0.,
    0.,
    -130632.6573424,
    199008.2507695,
    222878.4756832,
    0.,
    0.,
    0.,
    0.,
    8032.504691472,
    -2143.593577595,
    0.,
    0.,
    0.,
    -8032.504691472,
    0.,
    7133.754559318,
    0.,
    0.,
    0.,
    2143.593577595,
    -7133.754559318,
    0.  // QP 7
};

constexpr std::array<double, kNumQPs * 6 * 6> kPuu = {
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    5259.878305537,
    9932.579075823,
    3402.631809939,
    0.,
    0.,
    0.,
    -24476.28810745,
    3945.69190817,
    14331.70051426,
    0.,
    0.,
    0.,
    -72243.1983242,
    -50482.20381259,
    -9205.570213707,
    0.,
    0.,
    0.,  // QP 1
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    -22454.24793324,
    10599.75601088,
    62.82457748333,
    0.,
    0.,
    0.,
    -151208.9545336,
    24973.00181021,
    3339.00202713,
    0.,
    0.,
    0.,
    -68331.25525314,
    -44427.76893969,
    -2518.753876971,
    0.,
    0.,
    0.,  // QP 2
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    -4477.215986944,
    2320.228181973,
    922.5800439328,
    0.,
    0.,
    0.,
    -313742.8073858,
    -5534.800252915,
    64113.56542481,
    0.,
    0.,
    0.,
    -84956.030414,
    -54414.69222592,
    10012.01623986,
    0.,
    0.,
    0.,  // QP 3
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    57146.05831657,
    8907.36195072,
    -30181.83797779,
    0.,
    0.,
    0.,
    -371914.8053116,
    -109165.1287529,
    167157.4368918,
    0.,
    0.,
    0.,
    -147555.5970777,
    -111388.4238159,
    52019.0704363,
    0.,
    0.,
    0.,  // QP 4
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    77183.993751,
    24818.05123915,
    -75995.34009323,
    0.,
    0.,
    0.,
    -298162.4525991,
    -206798.6434849,
    219172.5050552,
    0.,
    0.,
    0.,
    -208756.3629627,
    -248402.3814645,
    129614.6497339,
    0.,
    0.,
    0.,  // QP 5
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    51002.97035598,
    6124.942615735,
    -108235.4341767,
    0.,
    0.,
    0.,
    -191403.9721169,
    -246041.6052807,
    216147.6081407,
    0.,
    0.,
    0.,
    -202368.6385284,
    -423499.5315683,
    195038.6349247,
    0.,
    0.,
    0.,  // QP 6
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    26641.25345202,
    -30784.92940852,
    -130632.6573424,
    0.,
    0.,
    0.,
    -123293.0646233,
    -249519.7291353,
    199008.2507695,
    0.,
    0.,
    0.,
    -155696.9812347,
    -547946.4435877,
    222878.4756832,
    0.,
    0.,
    0.  // QP 7
};

constexpr std::array<double, kNumQPs * 6 * 6> kQuu = {
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    2704.533593359,
    5045.754713661,
    -11271.05939004,
    0.,
    0.,
    0.,
    2368.612570317,
    17785.93033178,
    3307.618996744,
    0.,
    0.,
    0.,
    -11094.92632106,
    3720.671154735,
    70314.11063997,  // QP 1
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    1263.206625923,
    3626.925527762,
    -5754.601886892,
    0.,
    0.,
    0.,
    2712.043388638,
    21198.48684974,
    5525.137585753,
    0.,
    0.,
    0.,
    -6090.664775265,
    5700.435818024,
    58659.81418211,  // QP 2
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    89.98151665249,
    1323.732230709,
    -1376.079408024,
    0.,
    0.,
    0.,
    1109.032541408,
    77147.82471382,
    13110.3800426,
    0.,
    0.,
    0.,
    -2352.469996167,
    13137.76983773,
    59577.51532286,  // QP 3
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    5122.808384368,
    -23765.9281993,
    -1436.389909836,
    0.,
    0.,
    0.,
    -27547.63022841,
    148302.035447,
    43702.07609312,
    0.,
    0.,
    0.,
    1569.539714034,
    43973.57587716,
    89504.82897255,  // QP 4
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    24140.8944254,
    -47084.08515927,
    -6825.477557383,
    0.,
    0.,
    0.,
    -63344.39216072,
    170092.2312513,
    95184.29761274,
    0.,
    0.,
    0.,
    3831.896452589,
    96436.07611866,
    171424.6374178,  // QP 5
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    51097.43724235,
    -49794.11809666,
    93.74242546795,
    0.,
    0.,
    0.,
    -89950.16961486,
    159620.3991472,
    137404.5580242,
    0.,
    0.,
    0.,
    18988.06491057,
    137602.5740271,
    298452.3374463,  // QP 6
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    78211.77182302,
    -44670.96907162,
    17433.98293775,
    0.,
    0.,
    0.,
    -109943.6252315,
    146709.0717483,
    159880.1556629,
    0.,
    0.,
    0.,
    41859.8724774,
    156270.8432818,
    405244.0727313  // QP 7
};

constexpr std::array<double, kNumQPs * 6 * 6> kCuu = {
    1.252405841673e6,
    -316160.8714634,
    190249.1405163,
    0.,
    0.,
    0.,
    -316160.8714634,
    174443.0134269,
    -51312.45787663,
    0.,
    0.,
    0.,
    190249.1405163,
    -51312.45787663,
    68661.14489973,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    31406.91734868,
    25085.26404619,
    -17572.20634725,
    0.,
    0.,
    0.,
    25085.26404619,
    47547.80267146,
    6533.599954035,
    0.,
    0.,
    0.,
    -17572.20634725,
    6533.599954035,
    138595.2799799,  // QP 1
    1.33769634869e6,
    -195099.0548559,
    2748.101024543,
    0.,
    0.,
    0.,
    -195099.0548559,
    118918.450411,
    1945.323843732,
    0.,
    0.,
    0.,
    2748.101024543,
    1945.323843732,
    38895.20089869,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    23338.41787947,
    23236.72109144,
    -399.3549703606,
    0.,
    0.,
    0.,
    23236.72109144,
    52958.43856145,
    -4206.149649468,
    0.,
    0.,
    0.,
    -399.3549703606,
    -4206.149649468,
    141253.1435591,  // QP 2
    1.312971134329e6,
    52534.62528701,
    -259932.447777,
    0.,
    0.,
    0.,
    52534.62528701,
    89741.68392217,
    -3761.384183944,
    0.,
    0.,
    0.,
    -259932.447777,
    -3761.384183944,
    92797.1817487,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    21567.6848023,
    13022.03513926,
    26670.78525805,
    0.,
    0.,
    0.,
    13022.03513926,
    62149.59607555,
    -14432.23654511,
    0.,
    0.,
    0.,
    26670.78525805,
    -14432.23654511,
    133832.7191222,  // QP 3
    1.081138394585e6,
    333969.8032786,
    -430527.5506812,
    0.,
    0.,
    0.,
    333969.8032786,
    195547.1200349,
    -127423.0126465,
    0.,
    0.,
    0.,
    -430527.5506812,
    -127423.0126465,
    218824.4853804,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    34912.64847921,
    -3937.460207125,
    50123.55467118,
    0.,
    0.,
    0.,
    -3937.460207125,
    68219.92227345,
    -14095.63258756,
    0.,
    0.,
    0.,
    50123.55467118,
    -14095.63258756,
    114417.4292473,  // QP 4
    748350.9541536,
    497410.374804,
    -430244.2882178,
    0.,
    0.,
    0.,
    497410.374804,
    444679.9184161,
    -288893.1038727,
    0.,
    0.,
    0.,
    -430244.2882178,
    -288893.1038727,
    302479.1274303,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    57316.70645201,
    -17542.14397151,
    60066.47305368,
    0.,
    0.,
    0.,
    -17542.14397151,
    65470.47260674,
    -4142.791748881,
    0.,
    0.,
    0.,
    60066.47305368,
    -4142.791748881,
    94762.82094124,  // QP 5
    477154.2223662,
    511823.4458372,
    -341166.0554636,
    0.,
    0.,
    0.,
    511823.4458372,
    711671.9456551,
    -384910.8947082,
    0.,
    0.,
    0.,
    -341166.0554636,
    -384910.8947082,
    306683.8319788,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    76662.67143837,
    -22628.2146535,
    59832.70508077,
    0.,
    0.,
    0.,
    -22628.2146535,
    57325.69909916,
    6610.484964292,
    0.,
    0.,
    0.,
    59832.70508077,
    6610.484964292,
    83561.62946247,  // QP 6
    332963.4894648,
    466877.1452554,
    -266818.3173693,
    0.,
    0.,
    0.,
    466877.1452554,
    880186.2152254,
    -410842.761438,
    0.,
    0.,
    0.,
    -266818.3173693,
    -410842.761438,
    282360.2953098,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    87224.22496298,
    -22602.29984791,
    57195.84519625,
    0.,
    0.,
    0.,
    -22602.29984791,
    50635.7327372,
    12851.4255585,
    0.,
    0.,
    0.,
    57195.84519625,
    12851.4255585,
    79690.04229982  // QP 7
};

constexpr std::array<double, kNumNodes * kNumNodes * 6 * 6> kExpectedStiffnessMatrix_data = {
    // Node 1, Node 1
    569689.3362162, -38777.68238139, -34112.74014376, 4530.690247065, 69230.6811403, 36866.75913868,

    -38777.68238139, 68359.65943488, -16123.79317264, -4218.411719474, -4691.329984101,
    26527.28288064,

    -34112.74014376, -16123.79317264, 38260.55687584, -140.8938187951, -10291.07490605,
    160.6509943715,

    4530.690055812, -4218.408189901, -140.8763790592, 14431.53940242, 8876.843476231, 2709.854284646,

    69230.6771762, -4691.344342424, -10291.09843195, 7813.818461582, 46661.37532192, 1614.278886509,

    36866.76265972, 26527.28934502, 160.6542866115, -4312.726473091, 6566.435606407, 108524.4614643,

    // Node 1, Node 2
    -605850.7269137, 49685.73334359, 18151.21548352, -7850.011270668, 194879.3620557, 66623.21375896,

    49685.73334359, -104946.6662197, 29867.24691747, -5110.147228266, 17723.90298828, 38303.23407207,

    18151.21548352, 29867.24691747, -43584.50119038, 4204.881553587, -53104.87095801,
    -9873.994553907,

    -144.60021717, 6290.533848681, -7045.871909592, -20416.38905688, -2188.229101942,
    -280.7627537373,

    -93287.76652665, -11892.82169391, 27784.89779209, 5276.019621435, -29228.16003719,
    -13340.57095745,

    -59007.68631205, -57558.02068618, 12037.42191108, -11608.82467861, -8006.906619245,
    -69206.16193256,

    // Node 1, Node 3
    36161.39069751, -10908.0509622, 15961.52466024, 1956.096799256, -30149.77405728, -21205.59496411,

    -10908.0509622, 36587.00678477, -13743.45374483, -1148.716748667, -21415.13350213,
    -49376.01634631,

    15961.52466024, -13743.45374483, 5323.94431454, -10912.56352414, 20735.01930006, 19458.94672096,

    -4386.089838642, -2072.12565878, 7186.748288651, -829.2913309604, 6084.061735051,
    -889.8322537546,

    24057.08935045, 16584.16603633, -17493.79936014, 7869.022652151, -18264.56908122,
    -10933.24416261,

    22140.92365233, 31030.73134116, -12198.07619769, -963.4375102559, -12867.21632869,
    -25692.33475772,

    // Node 2, Node 1
    -605850.7269137, 49685.73334359, 18151.21548352, -144.6343818921, -93287.7466406,
    -59007.67630916,

    49685.73334359, -104946.6662197, 29867.24691747, 6290.538225767, -11892.86701094,
    -57557.99601899,

    18151.21548352, 29867.24691747, -43584.50119038, -7045.836401985, 27784.89698151, 12037.39855653,

    -7849.977105946, -5110.151605352, 4204.846045979, -20416.38905688, 894.4657108833,
    -9342.842314995,

    194879.3421696, 17723.94830531, -53104.87014743, 2193.32480861, -29228.16003719, -7325.429747877,

    66623.20375607, 38303.20940489, -9873.97119936, -2546.745117358, -14022.04782882,
    -69206.16193256,

    // Node 2, Node 2
    883410.7235792, 111663.4177865, -133962.585991, -35377.65212148, 5881.143872326, 59111.86946021,

    111663.4177865, 370419.6465568, -167395.5772476, -3996.045516334, 108888.2318195, 133129.5569735,

    -133962.585991, -167395.5772476, 157220.2611026, 43401.70962382, -84380.25059249,
    -73509.63004423,

    -35377.91914348, -3995.937572879, 43402.24249002, 79168.26100116, -69681.78029201,
    14134.49250439,

    5881.181750314, 108887.6849028, -84380.59929693, -88163.3303642, 424986.267748, 154807.1339731,

    59112.01659453, 133129.8093569, -73509.76575935, 32119.44497943, 157730.8368347, 438246.4304953,

    // Node 2, Node 3
    -277559.9966655, -161349.1511301, 115811.3705075, -25513.31260404, 123539.6672182,
    114378.3781508,

    -161349.1511301, -265472.9803372, 137528.3303301, 2596.893831142, 140104.421777, 264068.9453965,

    115811.3705075, 137528.3303301, -113635.7599122, 65351.33983397, -125130.3760381, -114590.646409,

    43227.89624942, 9106.089178231, -47607.088536, -7195.144647803, -25661.7728439, -19403.18480375,

    -200760.5239199, -126611.6332081, 137485.4694444, -33019.03158207, 57805.91444656,
    58040.06866028,

    -125735.2203506, -171433.0187618, 83383.73695871, -14981.13609737, 66668.5974604, 87549.27246588,

    // Node 3, Node 1
    36161.39069751, -10908.0509622, 15961.52466024, -4386.117052066, 24057.1014808, 22140.93353407,

    -10908.0509622, 36587.00678477, -13743.45374483, -2072.119772437, 16584.1176549, 31030.75715535,

    15961.52466024, -13743.45374483, 5323.94431454, 7186.794080991, -17493.82343661, -12198.09058474,

    1956.12401268, -1148.72263501, -10912.60931648, -829.2913309604, 4661.577812944, 762.6807791983,

    -30149.78618763, -21415.08512069, 20735.04337653, 9291.506574259, -18264.56908122,
    -12810.56006716,

    -21205.60484585, -49376.04216051, 19458.96110801, -2615.950543209, -10989.90042413,
    -25692.33475772,

    // Node 3, Node 2
    -277559.9966655, -161349.1511301, 115811.3705075, 43228.03926784, -200760.5923274,
    -125735.2698805,

    -161349.1511301, -265472.9803372, 137528.3303301, 9106.061255771, -126611.3943654,
    -171433.1466858,

    115811.3705075, 137528.3303301, -113635.7599122, -47607.30721297, 137485.5665608, 83383.81786146,

    -25513.45562246, 2596.921753602, 65351.55851094, -7195.144647803, -15807.69831469,
    -24151.5916188,

    123539.7356257, 140104.1829342, -125130.4731546, -42873.10611128, 57805.91444656, 65760.49554293,

    114378.4276807, 264069.0733205, -114590.7273117, -10232.72928231, 58948.17057775, 87549.27246588,

    // Node 3, Node 3
    241398.605968, 172257.2020923, -131772.8951677, 23557.61722567, -93390.1148328, -93172.89664176,

    172257.2020923, 228885.9735524, -123784.8765853, -1448.230559337, -118688.6403406,
    -214693.2939561,

    -131772.8951677, -123784.8765853, 108311.8155976, -54439.3804373, 104395.6129192, 95131.91939576,

    23557.33160978, -1448.199118592, -54438.94919445, 75656.15850236, -41527.99383711, 25853.1006121,

    -93389.94943804, -118689.0978135, 104395.4297781, -89794.34580364, 151830.0050476,
    122322.5566496,

    -93172.82283483, -214693.03116, 95131.76620372, 45440.96931327, 114661.6391408, 311150.0817408
};
const Kokkos::View<double[kNumNodes][kNumNodes][6][6], Kokkos::HostSpace>::const_type
    kExpectedStiffnessMatrix(kExpectedStiffnessMatrix_data.data());

}  // namespace openturbine::tests::curved_beam
