#include <string>
#include <vector>

// Assembly
struct Assembly {
    std::string turbine_class;      // IEC wind class of the wind turbine. The options are "I", "II",
                                    // "III", and 'IV'
    std::string turbulence_class;   // IEC turbulence class of the wind turbine. The options are "A",
                                    // "B", and "C"
    std::string drivetrain;         // String characterizing the drivetrain configuration
    std::string rotor_orientation;  // Orientation of the horizontal-axis rotor. The options are
                                    // "Upwind" and "Downwind"
    int number_of_blades;           // Number of blades of the rotor
    double rotor_diameter;  // Diameter of the rotor, defined as two times the blade length plus the
                            // hub diameter
    double hub_height;      // Height of the hub center over the ground (land-based) or the mean sea
                            // level (offshore)
    double rated_power;  // Nameplate power of the turbine, i.e. the rated electrical output of the
                         // generator.
    double lifetime;     // Turbine design lifetime in years.

    void parse(const YAML::Node& node) {
        turbine_class = node["turbine_class"] ? node["turbine_class"].as<std::string>() : "";
        turbulence_class =
            node["turbulence_class"] ? node["turbulence_class"].as<std::string>() : "";
        drivetrain = node["drivetrain"] ? node["drivetrain"].as<std::string>() : "";
        rotor_orientation =
            node["rotor_orientation"] ? node["rotor_orientation"].as<std::string>() : "";
        number_of_blades = node["number_of_blades"] ? node["number_of_blades"].as<int>() : 0;
        rotor_diameter = node["rotor_diameter"] ? node["rotor_diameter"].as<double>() : 0.;
        hub_height = node["hub_height"] ? node["hub_height"].as<double>() : 0.;
        rated_power = node["rated_power"] ? node["rated_power"].as<double>() : 0.;
        lifetime = node["lifetime"] ? node["lifetime"].as<double>() : 0.;
    }
};

// AirfoilPosition
struct AirfoilPosition {
    std::vector<double> grid;
    std::vector<std::string> labels;
};

// Chord
struct Chord {
    std::vector<double> grid;
    std::vector<double> values;
};

// Twist
struct Twist {
    std::vector<double> grid;
    std::vector<double> values;
};

// PitchAxis
struct PitchAxis {
    std::vector<double> grid;
    std::vector<double> values;
};

// TDividedByC
struct TDividedByC {
    std::vector<double> grid;
    std::vector<double> values;
};

// LDividedByD
struct LDividedByD {
    std::vector<double> grid;
    std::vector<double> values;
};

// CD
struct CD {
    std::vector<double> grid;
    std::vector<double> values;
};

// StallMargin
struct StallMargin {
    std::vector<double> grid;
    std::vector<double> values;
};

// X
struct X {
    std::vector<double> grid;
    std::vector<double> values;
};

// Y
struct Y {
    std::vector<double> grid;
    std::vector<double> values;
};

// Z
struct Z {
    std::vector<double> grid;
    std::vector<double> values;
};

// The reference system is located at blade root, with z aligned with the pitch axis, x pointing
// towards the suction sides of the airfoils (standard prebend will be negative) and y pointing to
// the trailing edge (standard sweep will be positive)
struct ReferenceAxis {
    X x;
    Y y;
    Z z;
};

// Rthick
struct Rthick {
    std::vector<double> grid;
    std::vector<double> values;
};

// OuterShapeBem
struct OuterShapeBem {
    AirfoilPosition airfoil_position;
    Chord chord;
    Twist twist;
    PitchAxis pitch_axis;
    TDividedByC t_divided_by_c;
    LDividedByD L_divided_by_D;
    CD c_d;
    StallMargin stall_margin;
    ReferenceAxis reference_axis;
    Rthick rthick;
};

// A
struct A {
    std::vector<double> grid;
    std::vector<double> values;
};

// E
struct E {
    std::vector<double> grid;
    std::vector<double> values;
};

// G
struct G {
    std::vector<double> grid;
    std::vector<double> values;
};

// IX
struct IX {
    std::vector<double> grid;
    std::vector<double> values;
};

// IY
struct IY {
    std::vector<double> grid;
    std::vector<double> values;
};

// K
struct K {
    std::vector<double> grid;
    std::vector<double> values;
};

// Dm
struct Dm {
    std::vector<double> grid;
    std::vector<double> values;
};

// KX
struct KX {
    std::vector<double> grid;
    std::vector<double> values;
};

// KY
struct KY {
    std::vector<double> grid;
    std::vector<double> values;
};

// Pitch
struct Pitch {
    std::vector<double> grid;
    std::vector<double> values;
};

// RiX
struct RiX {
    std::vector<double> grid;
    std::vector<double> values;
};

// RiY
struct RiY {
    std::vector<double> grid;
    std::vector<double> values;
};

// XCg
struct XCg {
    std::vector<double> grid;
    std::vector<double> values;
};

// XE
struct XE {
    std::vector<double> grid;
    std::vector<double> values;
};

// XSh
struct XSh {
    std::vector<double> grid;
    std::vector<double> values;
};

// YCg
struct YCg {
    std::vector<double> grid;
    std::vector<double> values;
};

// YE
struct YE {
    std::vector<double> grid;
    std::vector<double> values;
};

// YSh
struct YSh {
    std::vector<double> grid;
    std::vector<double> values;
};

// Timoschenko beam as in HAWC2
struct TimoschenkoHawc {
    ReferenceAxis reference_axis;
    A A;
    E E;
    G G;
    IX I_x;
    IY I_y;
    K K;
    Dm dm;
    KX k_x;
    KY k_y;
    Pitch pitch;
    RiX ri_x;
    RiY ri_y;
    XCg x_cg;
    XE x_e;
    XSh x_sh;
    YCg y_cg;
    YE y_e;
    YSh y_sh;
};

// T11
struct T11 {
    std::vector<double> grid;
    std::vector<double> values;
};

// T22
struct T22 {
    std::vector<double> grid;
    std::vector<double> values;
};

// Ea
struct Ea {
    std::vector<double> grid;
    std::vector<double> values;
};

// E11
struct E11 {
    std::vector<double> grid;
    std::vector<double> values;
};

// E22
struct E22 {
    std::vector<double> grid;
    std::vector<double> values;
};

// Gj
struct Gj {
    std::vector<double> grid;
    std::vector<double> values;
};

// XCe
struct XCe {
    std::vector<double> grid;
    std::vector<double> values;
};

// YCe
struct YCe {
    std::vector<double> grid;
    std::vector<double> values;
};

// DeltaTheta
struct DeltaTheta {
    std::vector<double> grid;
    std::vector<double> values;
};

// J1
struct J1 {
    std::vector<double> grid;
    std::vector<double> values;
};

// J2
struct J2 {
    std::vector<double> grid;
    std::vector<double> values;
};

// J3
struct J3 {
    std::vector<double> grid;
    std::vector<double> values;
};

// Geometrically exact beams with simplified properties
struct CpLambdaBeam {
    ReferenceAxis reference_axis;
    T11 T11;
    T22 T22;
    Ea EA;
    E11 E11;
    E22 E22;
    Gj GJ;
    XCe x_ce;
    YCe y_ce;
    Dm dm;
    DeltaTheta delta_theta;
    XSh x_sh;
    YSh y_sh;
    J1 J1;
    J2 J2;
    J3 J3;
    XCg x_cg;
    YCg y_cg;
};

// StiffMatrix
struct StiffMatrix {
    std::vector<double> grid;
    std::vector<std::vector<double>> values;
};

// SixXSix
struct SixXSix {
    ReferenceAxis reference_axis;
    StiffMatrix stiff_matrix;
};

// ElasticPropertiesMb
struct ElasticPropertiesMb {
    TimoschenkoHawc timoschenko_hawc;
    CpLambdaBeam cp_lambda_beam;
    SixXSix six_x_six;
};

// Root
struct Root {
    double d_f;        // Diameter of the fastener, default is M30, so 0.03 meters
    double sigma_max;  // Max stress on bolt
};

// non-dimensional location of the point along the non-dimensional arc length
struct StartNdArc {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed;  // Name of the layer to lock the edge
};

// non-dimensional location of the point along the non-dimensional arc length
struct EndNdArc {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed;  // Name of the layer to lock the edge
};

// rotation of the chord axis around the pitch axis
struct Rotation {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed;  // Name of the layer to lock the edge
};

// dimensional offset in respect to the pitch axis along the x axis, which is the chord line rotated
// by a user-defined angle. Negative values move the midpoint towards the leading edge, positive
// towards the trailing edge
struct OffsetYPa {
    std::vector<double> grid;
    std::vector<double> values;
};

// Webs
struct Webs {
    std::string name;  // structural component identifier
    StartNdArc start_nd_arc;
    EndNdArc end_nd_arc;
    Rotation rotation;
    OffsetYPa offset_y_pa;
};

// thickness of the laminate
struct Thickness {
    std::vector<double> grid;
    std::vector<double> values;
};

// number of plies of the laminate
struct NPlies {
    std::vector<double> grid;
    std::vector<double> values;
};

// orientation of the fibers
struct FiberOrientation {
    std::vector<double> grid;
    std::vector<double> values;
};

// dimensional width of the component along the arc
struct Width {
    std::vector<double> grid;
    std::vector<double> values;
};

// non-dimensional location of the point along the non-dimensional arc length
struct MidpointNdArc {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed;  // Name of the layer to lock the edge
};

// Layers
struct Layers {
    std::string name;      // structural component identifier
    std::string material;  // material identifier
    std::string web;  // web to which the layer is associated to, only to be defined for web layers
    Thickness thickness;                 // thickness of the laminate
    NPlies n_plies;                      // number of plies of the laminate
    FiberOrientation fiber_orientation;  // orientation of the fibers
    Width width;                         // dimensional width of the component along the arc
    MidpointNdArc midpoint_nd_arc;
    StartNdArc start_nd_arc;
    EndNdArc end_nd_arc;
    Rotation rotation;
    OffsetYPa offset_y_pa;
};

// This is a spanwise joint along the blade, usually adopted to ease transportation constraints.
// WISDEM currently supports a single joint.
struct Joint {
    double position;          // Spanwise position of the segmentation joint.
    double mass;              // Mass of the joint.
    double cost;              // Cost of the joint.
    std::string bolt;         // Bolt size for the blade bolted joint
    double nonmaterial_cost;  // Cost of the joint not from materials.
    std::string
        reinforcement_layer_ss;  // Layer identifier for the joint reinforcement on the suction side
    std::string
        reinforcement_layer_ps;  // Layer identifier for the joint reinforcement on the pressure side
};

// InternalStructure2DFem
struct InternalStructure2DFem {
    Root root;
    ReferenceAxis reference_axis;
    std::vector<Webs> webs;      // ...
    std::vector<Layers> layers;  // ...
    Joint joint;  // This is a spanwise joint along the blade, usually adopted to ease transportation
                  // constraints. WISDEM currently supports a single joint.
};

// Blade
struct Blade {
    OuterShapeBem outer_shape_bem;
    ElasticPropertiesMb elastic_properties_mb;
    InternalStructure2DFem internal_structure_2d_fem;
};

// OuterShapeBem_1
struct OuterShapeBem_1 {
    double diameter;    // Diameter of the hub measured at the blade root positions.
    double cone_angle;  // Rotor precone angle, defined positive for both upwind and downwind rotors.
    double drag_coefficient;  // Equivalent drag coefficient to compute the aerodynamic forces
                              // generated on the hub.
};

// ElasticPropertiesMb_1
struct ElasticPropertiesMb_1 {
    double system_mass;  // Mass of the hub system, which includes the hub, the spinner, the blade
                         // bearings, the pitch actuators, the cabling, ....
    std::vector<double>
        system_inertia;  // Inertia of the hub system, on the hub reference system, which has the x
                         // aligned with the rotor axis, and y and z perpendicular to it.
    std::vector<double> system_center_mass;  // Center of mass of the hub system. Work in progress.
};

// Hub
struct Hub {
    OuterShapeBem_1 outer_shape_bem;
    ElasticPropertiesMb_1 elastic_properties_mb;
};

// User input override of generator rpm-efficiency values, with rpm as grid input and eff as values
// input
struct GeneratorRpmEfficiencyUser {
    std::vector<double> grid;
    std::vector<double> values;
};

// Thickness of the hollow elliptical bedplate used in direct drive configurations
struct BedplateWallThickness {
    std::vector<double> grid;
    std::vector<double> values;
};

// Inputs to WISDEM specific drivetrain sizing tool, DrivetrainSE
struct Drivetrain {
    double uptilt;                 // Tilt angle of the nacelle, always defined positive.
    double distance_tt_hub;        // Vertical distance between the tower top and the hub center.
    double distance_hub_mb;        // Distance from hub flange to first main bearing along shaft.
    double distance_mb_mb;         // Distance from first to second main bearing along shaft.
    double overhang;               // Horizontal distance between the tower axis and the rotor apex.
    double generator_length;       // Length of generator along the shaft
    double generator_radius_user;  // User input override of generator radius, only used when using
                                   // simple generator scaling
    double generator_mass_user;    // User input override of generator mass, only used when using
                                   // simple generator mass scaling
    GeneratorRpmEfficiencyUser
        generator_rpm_efficiency_user;  // User input override of generator rpm-efficiency values,
                                        // with rpm as grid input and eff as values input
    double gear_ratio;  // Gear ratio of the drivetrain. Set it to 1 for direct drive machines.
    double gearbox_length_user;  // User input override of gearbox length along shaft, only used when
                                 // using gearbox_mass_user is > 0
    double gearbox_radius_user;  // User input override of gearbox radius, only used when using
                                 // gearbox_mass_user is > 0
    double gearbox_mass_user;    // User input override of gearbox mass
    double gearbox_efficiency;   // Efficiency of the gearbox system.
    double damping_ratio;        // Damping ratio for the drivetrain system
    std::vector<double> lss_diameter;        // Diameter of the low speed shaft at beginning
                                             // (generator/gearbox) and end (hub) points
    std::vector<double> lss_wall_thickness;  // Thickness of the low speed shaft at beginning
                                             // (generator/gearbox) and end (hub) points
    std::string lss_material;                // Material name identifier
    double hss_length;                       // Length of the high speed shaft
    std::vector<double> hss_diameter;  // Diameter of the high speed shaft at beginning (generator)
                                       // and end (generator) points
    std::vector<double> hss_wall_thickness;  // Thickness of the high speed shaft at beginning
                                             // (generator) and end (generator) points
    std::string hss_material;                // Material name identifier
    std::vector<double> nose_diameter;  // Diameter of the nose/turret at beginning (bedplate) and
                                        // end (main bearing) points
    std::vector<double> nose_wall_thickness;  // Thickness of the nose/turret at beginning (bedplate)
                                              // and end (main bearing) points
    BedplateWallThickness bedplate_wall_thickness;  // Thickness of the hollow elliptical bedplate
                                                    // used in direct drive configurations
    double bedplate_flange_width;  // Bedplate I-beam flange width used in geared configurations
    double
        bedplate_flange_thickness;  // Bedplate I-beam flange thickness used in geared configurations
    double bedplate_web_thickness;  // Bedplate I-beam web thickness used in geared configurations
    double brake_mass_user;  // Override regular regression-based calculation of brake mass with this
                             // value
    double hvac_mass_coefficient;  // Regression-based scaling coefficient on machine rating to get
                                   // HVAC system mass
    double converter_mass_user;    // Override regular regression-based calculation of converter mass
                                   // with this value
    double transformer_mass_user;  // Override regular regression-based calculation of transformer
                                   // mass with this value
    std::string bedplate_material;  // Material name identifier
    std::string mb1Type;            // Type of bearing for first main bearing
    std::string mb2Type;            // Type of bearing for second main bearing
    bool uptower;  // If power electronics are located uptower (True) or at tower base (False)
    std::string gear_configuration;   // 3-letter string of Es or Ps to denote epicyclic or parallel
                                      // gear configuration
    std::vector<int> planet_numbers;  // Number of planets for epicyclic stages (use 0 for parallel)
};

// Generator
struct Generator {
    double mass_coefficient;  // When not doing a detailed generator design, use a simplified
                              // approach to generator scaling. This input allows for overriding of
                              // the regression-based scaling coefficient to obtain generator mass
    std::string generator_type;
    double B_r;          // Words
    double P_Fe0e;       // Words
    double P_Fe0h;       // Words
    double S_N;          // Words
    double S_Nmax;       // Words
    double alpha_p;      // Words
    double b_r_tau_r;    // Words
    double b_ro;         // Words
    double b_s_tau_s;    // Words
    double b_so;         // Words
    double cofi;         // Words
    double freq;         // Words
    double h_i;          // Words
    double h_sy0;        // Words
    double h_w;          // Words
    double k_fes;        // Words
    double k_fillr;      // Words
    double k_fills;      // Words
    double k_s;          // Words
    int m;               // Words
    double mu_0;         // Permittivity of free space
    double mu_r;         // Words
    double p;            // Words
    double phi;          // Words
    int q1;              // Words
    int q3;              // Words
    double ratio_mw2pp;  // Words
    double resist_Cu;    // Resistivity of copper
    double sigma;        // Maximum allowable shear stress
    double y_tau_p;      // Words
    double y_tau_pr;     // Words
    double I_0;          // Words
    double d_r;          // Words
    double h_m;          // Words
    double h_0;          // Words
    double h_s;          // Words
    double len_s;        // Words
    double n_r;          // Words
    double rad_ag;       // Words
    double t_wr;         // Words
    double n_s;          // Words
    double b_st;         // Words
    double d_s;          // Words
    double t_ws;         // Words
    double rho_Copper;   // Copper density
    double rho_Fe;       // Structural steel density
    double rho_Fes;      // Electrical steel density
    double rho_PM;       // Permanent magnet density
    double C_Cu;         // Copper cost
    double C_Fe;         // Structural steel cost
    double C_Fes;        // Electrical steel cost
    double C_PM;         // Permanent magnet cost
};

// Nacelle
struct Nacelle {
    Drivetrain drivetrain;  // Inputs to WISDEM specific drivetrain sizing tool, DrivetrainSE
    Generator generator;
};

// OuterDiameter
struct OuterDiameter {
    std::vector<double> grid;
    std::vector<double> values;
};

// DragCoefficient
struct DragCoefficient {
    std::vector<double> grid;
    std::vector<double> values;
};

// OuterShapeBem_2
struct OuterShapeBem_2 {
    ReferenceAxis reference_axis;
    OuterDiameter outer_diameter;
    DragCoefficient drag_coefficient;
};

// InternalStructure2DFem_1
struct InternalStructure2DFem_1 {
    double outfitting_factor;  // Scaling factor for the tower mass to account for auxiliary
                               // structures, such as elevator, ladders, cables, platforms, etc
    ReferenceAxis reference_axis;
    std::vector<Layers> layers;  // ...
};

// Tower
struct Tower {
    OuterShapeBem_2 outer_shape_bem;
    ElasticPropertiesMb elastic_properties_mb;
    InternalStructure2DFem_1 internal_structure_2d_fem;
};

// OuterShape
struct OuterShape {
    ReferenceAxis reference_axis;
    OuterDiameter outer_diameter;
    DragCoefficient drag_coefficient;
};

// InternalStructure2DFem_2
struct InternalStructure2DFem_2 {
    double outfitting_factor;  // Scaling factor for the tower mass to account for auxiliary
                               // structures, such as elevator, ladders, cables, platforms, etc
    ReferenceAxis reference_axis;
    std::vector<Layers> layers;  // ...
};

// Monopile
struct Monopile {
    double transition_piece_mass;    // Total mass of transition piece
    double transition_piece_cost;    // Total cost of transition piece
    double gravity_foundation_mass;  // Total mass of gravity foundation addition onto monopile
    OuterShape outer_shape;
    ElasticPropertiesMb elastic_properties_mb;
    InternalStructure2DFem_2 internal_structure_2d_fem;
};

// Jacket
struct Jacket {
    double transition_piece_mass;    // Total mass of transition piece
    double transition_piece_cost;    // Total cost of transition piece
    double gravity_foundation_mass;  // Total mass of gravity foundation addition onto monopile
    std::string material;            // Material of jacket members
    int n_bays;            // Number of bays (x-joints) in the vertical direction for jackets.
    int n_legs;            // Number of legs for jacket.
    double r_foot;         // Radius of foot (bottom) of jacket, in meters.
    double r_head;         // Radius of head (top) of jacket, in meters.
    double height;         // Overall jacket height, meters.
    double leg_thickness;  // Leg thickness, meters. Constant throughout each leg.
    std::vector<double> brace_diameters;
    std::vector<double> brace_thicknesses;
    std::vector<double> bay_spacing;
    std::vector<double> leg_spacing;
    bool x_mb;            // Mud brace included if true.
    double leg_diameter;  // Leg diameter, meters. Constant throughout each leg.
};

// If this joint is compliant is certain DOFs, then specify which are compliant (True) in the
// member/element coordinate system).  If not specified, default is all entries are False (completely
// rigid).  For instance, a ball joint would be Rx=Ry=Rz=False, Rxx=Ryy=Rzz=True
struct Reactions {
    bool Rx;
    bool Ry;
    bool Rz;
    bool Rxx;
    bool Ryy;
    bool Rzz;
    std::vector<double> Euler;  // Euler angles [alpha, beta, gamma] that describe the rotation of
                                // the Reaction coordinate system relative to the global coordinate
                                // system α is a rotation around the z axis, β is a rotation around
                                // the x' axis, γ is a rotation around the z" axis.
};

// Joints
struct Joints {
    std::string name;  // Unique name of the joint (node)
    std::vector<double>
        location;      // Coordinates (x,y,z or r,θ,z) of the joint in the global coordinate system.
    bool transition;   // Whether the transition piece and turbine tower attach at this node
    bool cylindrical;  // Whether to use cylindrical coordinates (r,θ,z), with (r,θ) lying in the
                       // x/y-plane, instead of Cartesian coordinates.
    Reactions reactions;  // If this joint is compliant is certain DOFs, then specify which are
                          // compliant (True) in the member/element coordinate system).  If not
                          // specified, default is all entries are False (completely rigid).  For
                          // instance, a ball joint would be Rx=Ry=Rz=False, Rxx=Ryy=Rzz=True
};

// OuterShape_1
struct OuterShape_1 {
    std::string shape;  // Specifies cross-sectional shape of the member.  If circular, then the
                        // outer_diameter field is required.  If polygonal, then the side_lengths,
                        // angles, and rotation fields are required
    OuterDiameter outer_diameter;  // Gridded values describing diameter at non-dimensional axis from
                                   // joint1 to joint2
    std::vector<double> side_lengths1;  // Polygon side lengths at joint1
    std::vector<double> side_lengths2;  // Polygon side lengths at joint1
    std::vector<double> angles;  // Polygon angles with the ordering such that angle[i] is between
                                 // side_length[i] and side_length[i+1]
    double rotation;  // Angle between principle axes of the cross-section and the member coordinate
                      // system.  Essentially the rotation of the member if both joints were placed
                      // on the global x-y axis with the first side length along the z-axis
};

// Layers_1
struct Layers_1 {
    std::string name;      // structural component identifier
    std::string material;  // material identifier
    Thickness thickness;   // Gridded values describing thickness along non-dimensional axis from
                           // joint1 to joint2
};

// RingStiffeners
struct RingStiffeners {
    std::string material;  // material identifier
    double flange_thickness;
    double flange_width;
    double web_height;
    double web_thickness;
    double spacing;  // Spacing between stiffeners in non-dimensional grid coordinates. Value of 0.0
                     // means no stiffeners
};

// LongitudinalStiffeners
struct LongitudinalStiffeners {
    std::string material;  // material identifier
    double flange_thickness;
    double flange_width;
    double web_height;
    double web_thickness;
    double
        spacing;  // Spacing between stiffeners in angle (radians). Value of 0.0 means no stiffeners
};

// Bulkhead
struct Bulkhead {
    std::string material;  // material identifier
    Thickness
        thickness;  // thickness of the bulkhead at non-dimensional locations of the member [0..1]
};

// Ballast
struct Ballast {
    bool variable_flag;  // If true, then this ballast is variable and adjusted by control system. If
                         // false, then considered permanent
    std::string material;  // material identifier
    std::vector<double> grid;
    double volume;  // Total volume of ballast (permanent ballast only)
};

// InternalStructure
struct InternalStructure {
    double outfitting_factor;      // Scaling factor for the member mass to account for auxiliary
                                   // structures, such as elevator, ladders, cables, platforms,
                                   // fasteners, etc
    std::vector<Layers_1> layers;  // Material layer properties
    RingStiffeners ring_stiffeners;
    LongitudinalStiffeners longitudinal_stiffeners;
    Bulkhead bulkhead;
    std::vector<Ballast> ballast;  // Different types of permanent and/or variable ballast
};

// AxialJoints
struct AxialJoints {
    std::string name;  // Unique name of joint
    double grid;       // Non-dimensional value along member axis
};

// Members
struct Members {
    std::string name;    // Name of the member
    std::string joint1;  // Name of joint/node connection
    std::string joint2;  // Name of joint/node connection
    OuterShape_1 outer_shape;
    InternalStructure internal_structure;
    std::vector<AxialJoints>
        axial_joints;  // Define joints along non-dimensional axis of this member
    double Ca;         // User-defined added mass coefficient
    double Cp;         // User-defined pressure coefficient
    double Cd;         // User-defined drag coefficient
};

// RigidBodies
struct RigidBodies {
    std::string joint1;  // Name of joint/node connection
    double mass;         // Mass of this rigid body
    double cost;         // Cost of this rigid body
    std::vector<double>
        cm_offset;  // Offset from joint location to center of mass (CM) of body in dx, dy, dz
    std::vector<double> moments_of_inertia;  // Moments of inertia around body CM in Ixx, Iyy, Izz
    double Ca;                               // User-defined added mass coefficient
    double Cp;                               // User-defined pressure coefficient
    double Cd;                               // User-defined drag coefficient
};

// Ontology definition for floating platforms (substructures) suitable for use with the WEIS
// co-design analysis tool
struct FloatingPlatform {
    std::vector<Joints> joints;
    std::vector<Members> members;
    std::vector<RigidBodies> rigid_bodies;
    double transition_piece_mass;  // Total mass of transition piece
    double transition_piece_cost;  // Total cost of transition piece
};

// Nodes
struct Nodes {
    std::string name;  // Name or ID of this node for use in line segment
    std::string node_type;
    std::vector<double>
        location;  // – Coordinates x, y, and z of the connection (relative to inertial reference
                   // frame if Fixed or Connect, relative to platform reference frame if Vessel). In
                   // the case of Connect nodes, it is simply an initial guess for position before
                   // MoorDyn calculates the equilibrium initial position.
    std::string joint;  // For anchor positions and fairlead attachments, reference a joint name from
                        // the "joints" section or an "axial_joint" on a member
    std::string anchor_type;  // Name of anchor type from anchor_type list
    std::string fairlead_type;
    double node_mass;    // Clump weight mass
    double node_volume;  // Floater volume
    double drag_area;    // Product of drag coefficient and projected area (assumed constant in all
                         // directions) to calculate a drag force for the node
    double added_mass;  // Added mass coefficient used along with node volume to calculate added mass
                        // on node
};

// Lines
struct Lines {
    std::string name;           // ID of this line
    std::string line_type;      // Reference to line type database
    double unstretched_length;  // length of line segment prior to tensioning
    std::string node1;          // node id of first line connection
    std::string node2;          // node id of second line connection
};

// LineTypes
struct LineTypes {
    std::string name;  // Name of material or line type to be referenced by line segments
    double diameter;   // the volume-equivalent diameter of the line – the diameter of a cylinder
                       // having the same displacement per unit length
    std::string type;  // Type of material for property lookup
    double mass_density;  // mass per unit length (in air)
    double
        stiffness;  // axial line stiffness, product of elasticity modulus and cross-sectional area
    double cost;    // cost per unit length
    double breaking_load;          // line break tension
    double damping;                // internal damping (BA)
    double transverse_added_mass;  // transverse added mass coefficient (with respect to line
                                   // displacement)
    double tangential_added_mass;  // tangential added mass coefficient (with respect to line
                                   // displacement)
    double transverse_drag;        // transverse drag coefficient (with respect to frontal area, d*l)
    double tangential_drag;  // tangential drag coefficient (with respect to surface area, π*d*l)
};

// AnchorTypes
struct AnchorTypes {
    std::string name;         // Name of anchor to be referenced by anchor_id in Nodes section
    std::string type;         // Type of anchor for property lookup
    double mass;              // mass of the anchor
    double cost;              // cost of the anchor
    double max_lateral_load;  // Maximum lateral load (parallel to the sea floor) that the anchor can
                              // support
    double max_vertical_load;  // Maximum vertical load (perpendicular to the sea floor) that the
                               // anchor can support
};

// Ontology definition for mooring systems suitable for use with the WEIS co-design analysis tool
struct Mooring {
    std::vector<Nodes> nodes;           // List of nodes in the mooring system
    std::vector<Lines> lines;           // List of all mooring line properties in the mooring system
    std::vector<LineTypes> line_types;  // List of mooring line properties used in the system
    std::vector<AnchorTypes> anchor_types;  // List of anchor properties used in the system
};

// Components
struct Components {
    Blade blade;
    Hub hub;
    Nacelle nacelle;
    Tower tower;
    Monopile monopile;
    Jacket jacket;
    FloatingPlatform
        floating_platform;  // Ontology definition for floating platforms (substructures) suitable
                            // for use with the WEIS co-design analysis tool
    Mooring mooring;        // Ontology definition for mooring systems suitable for use with the WEIS
                            // co-design analysis tool
};

// Airfoil coordinates described from trailing edge (x=1) along the suction side (y>0) to leading
// edge (x=0) back to trailing edge (x=1) along the pressure side (y<0)
struct Coordinates {
    std::vector<double> x;
    std::vector<double> y;
};

// CL
struct CL {
    std::vector<double> grid;
    std::vector<double> values;
};

// CM
struct CM {
    std::vector<double> grid;
    std::vector<double> values;
};

// Lift, drag and moment coefficients expressed in terms of angles of attack
struct Polars {
    std::string configuration;  // Text to identify the setup for the definition of the polars
    double re;                  // Reynolds number of the polars
    CL c_l;
    CD c_d;
    CM c_m;
};

// Airfoils
struct Airfoils {
    std::string name;         // Name of the airfoil
    Coordinates coordinates;  // Airfoil coordinates described from trailing edge (x=1) along the
                              // suction side (y>0) to leading edge (x=0) back to trailing edge (x=1)
                              // along the pressure side (y<0)
    double relative_thickness;   // Thickness of the airfoil expressed non-dimensional
    double aerodynamic_center;   // Non-dimensional chordwise coordinate of the aerodynamic center
    std::vector<Polars> polars;  // Different sets of polars at varying conditions
};

// Materials
struct Materials {
    std::string name;         // Name of the material
    std::string description;  // Optional field describing the material
    std::string source;       // Optional field describing where the data come from
    int orth;                 // Flag to switch between isotropic (0) and orthotropic (1) materials
    double rho;  // Density of the material. For composites, this is the density of the laminate once
                 // cured
    double E;  // Stiffness modulus. For orthotropic materials, it consists of an array with E11, E22
               // and E33.
    double G;  // Shear stiffness modulus. For orthotropic materials, it consists of an array with
               // G12, G13 and G23
    double nu;  // Poisson ratio. For orthotropic materials, it consists of an array with nu12, nu13
                // and nu23. For isotropic materials, a minimum of -1 and a maximum of 0.5 are
                // imposed. No limits are imposed to anisotropic materials.
    double alpha;  // Thermal coefficient of expansion
    double Xt;  // Ultimate tensile strength. For orthotropic materials, it consists of an array with
                // the strength in directions 11, 22 and 33. The values must be positive
    double Xc;  // Ultimate compressive strength. For orthotropic materials, it consists of an array
                // with the strength in directions 11, 22 and 33. The values must be positive
    double Xy;  // Ultimate yield strength for metals. For orthotropic materials, it consists of an
                // array with the strength in directions 12, 13 and 23
    double S;   // Ultimate shear strength. For orthotropic materials, it consists of an array with
                // the strength in directions 12, 13 and 23
    double ply_t;      // Ply thickness of the composite material
    double unit_cost;  // Unit cost of the material. For composites, this is the unit cost of the dry
                       // fabric.
    double fvf;        // Fiber volume fraction of the composite material
    double fwf;        // Fiber weight fraction of the composite material
    double fiber_density;     // Density of the fibers of a composite material.
    double area_density_dry;  // Aerial density of a fabric of a composite material.
    int component_id;         // Flag used by the NREL blade cost model
                       // https://www.nrel.gov/docs/fy19osti/73585.pdf to define the manufacturing
                       // process behind the laminate. 0 - coating, 1 - sandwich filler , 2 - shell
                       // skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
    double waste;  // Fraction of material that ends up wasted during manufacturing. This quantity is
                   // used in the NREL blade cost model https://www.nrel.gov/docs/fy19osti/73585.pdf
    double roll_mass;  // Mass of a fabric roll. This quantity is used in the NREL blade cost model
                       // https://www.nrel.gov/docs/fy19osti/73585.pdf
    double GIc;   // Mode 1 critical energy-release rate. It is used by NuMAD from Sandia National
                  // Laboratories
    double GIIc;  // Mode 2 critical energy-release rate. It is used by NuMAD from Sandia National
                  // Laboratories
    double alp0;  // Fracture angle under pure transverse compression. It is used by NuMAD from
                  // Sandia National Laboratories
    double A;     // Fatigue S/N curve fitting parameter S=A*N^(-1/m)
    double m;     // Fatigue S/N curve fitting parameter S=A*N^(-1/m)
    double R;     // Fatigue stress ratio

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        description = node["description"] ? node["description"].as<std::string>() : "";
        source = node["source"] ? node["source"].as<std::string>() : "";
        orth = node["orth"] ? node["orth"].as<int>() : 0;
        rho = node["rho"] ? node["rho"].as<double>() : 0.;
        E = node["E"] ? node["E"].as<double>() : 0.;
        G = node["G"] ? node["G"].as<double>() : 0.;
        nu = node["nu"] ? node["nu"].as<double>() : 0.;
        alpha = node["alpha"] ? node["alpha"].as<double>() : 0.;
        Xt = node["Xt"] ? node["Xt"].as<double>() : 0.;
        Xc = node["Xc"] ? node["Xc"].as<double>() : 0.;
        Xy = node["Xy"] ? node["Xy"].as<double>() : 0.;
        S = node["S"] ? node["S"].as<double>() : 0.;
        ply_t = node["ply_t"] ? node["ply_t"].as<double>() : 0.;
        unit_cost = node["unit_cost"] ? node["unit_cost"].as<double>() : 0.;
        fvf = node["fvf"] ? node["fvf"].as<double>() : 0.;
        fwf = node["fwf"] ? node["fwf"].as<double>() : 0.;
        fiber_density = node["fiber_density"] ? node["fiber_density"].as<double>() : 0.;
        area_density_dry = node["area_density_dry"] ? node["area_density_dry"].as<double>() : 0.;
        component_id = node["component_id"] ? node["component_id"].as<int>() : 0;
        waste = node["waste"] ? node["waste"].as<double>() : 0.;
        roll_mass = node["roll_mass"] ? node["roll_mass"].as<double>() : 0.;
        GIc = node["GIc"] ? node["GIc"].as<double>() : 0.;
        GIIc = node["GIIc"] ? node["GIIc"].as<double>() : 0.;
        alp0 = node["alp0"] ? node["alp0"].as<double>() : 0.;
        A = node["A"] ? node["A"].as<double>() : 0.;
        m = node["m"] ? node["m"].as<double>() : 0.;
        R = node["R"] ? node["R"].as<double>() : 0.;
    }
};

// Supervisory
struct Supervisory {
    double Vin;    // Cut-in wind speed of the wind turbine.
    double Vout;   // Cut-out wind speed of the wind turbine.
    double maxTS;  // Maximum allowable blade tip speed.
};

// Pitch_1
struct Pitch_1 {
    double min_pitch;       // Minimum pitch angle, where the default is 0 degrees. It is used by the
                            // ROSCO controller (https://github.com/NREL/ROSCO)
    double max_pitch_rate;  // Maximum pitch rate of the rotor blades.
};

// Torque
struct Torque {
    double max_torque_rate;  // Maximum torque rate of the wind turbine generator.
    double tsr;  // Rated tip speed ratio of the wind turbine. As default, it is maintained constant
                 // in region II.
    double VS_minspd;  // Minimum rotor speed. It is used by the ROSCO controller
                       // (https://github.com/NREL/ROSCO)
    double VS_maxspd;  // Maximum rotor speed. It is used by the ROSCO controller
                       // (https://github.com/NREL/ROSCO)
};

// Control
struct Control {
    Supervisory supervisory;
    Pitch_1 pitch;
    Torque torque;
};

// Environment
struct Environment {
    double gravity;               // Gravitational acceleration
    double air_density;           // Density of air.
    double air_dyn_viscosity;     // Dynamic viscosity of air.
    double air_pressure;          // Atmospheric pressure of air
    double air_vapor_pressure;    // Vapor pressure of fluid
    double weib_shape_parameter;  // Shape factor of the Weibull wind distribution.
    double air_speed_sound;       // Speed of sound in air.
    double shear_exp;             // Shear exponent of the atmospheric boundary layer.
    double water_density;         // Density of water.
    double water_dyn_viscosity;   // Dynamic viscosity of water.
    double water_depth;           // Water depth for offshore environment.
    double soil_shear_modulus;    // Shear modulus of the soil.
    double soil_poisson;          // Poisson ratio of the soil.
    double V_mean;  // Average inflow wind speed. If different than 0, this will overwrite the V mean
                    // of the IEC wind class
};

// Bos
struct Bos {
    double plant_turbine_spacing;  // Distance between turbines in the primary grid streamwise
                                   // direction in rotor diameters
    double plant_row_spacing;  // Distance between turbine rows in the cross-wind direction in rotor
                               // diameters
    double commissioning_pct;  // Fraction of total BOS cost that is due to commissioning
    double decommissioning_pct;          // Fraction of total BOS cost that is due to decommissioning
    double distance_to_substation;       // Distance from centroid of plant to substation in km
    double distance_to_interconnection;  // Distance from substation to grid connection in km
    double distance_to_landfall;         // Distance from plant centroid to export cable landfall for
                                         // offshore plants
    double distance_to_site;             // Distance from port to plant centroid for offshore plants
    double interconnect_voltage;         // Voltage of cabling to grid interconnection
    double port_cost_per_month;          // Monthly port rental fees
    double site_auction_price;           // Cost to secure site lease
    double site_assessment_plan_cost;    // Cost to do engineering plan for site assessment
    double site_assessment_cost;         // Cost to execute site assessment
    double construction_operations_plan_cost;  // Cost to do construction planning
    double boem_review_cost;  // Cost for additional review by U.S. Dept of Interior Bureau of Ocean
                              // Energy Management (BOEM)
    double design_install_plan_cost;  // Cost to do installation planning
};

// Costs
struct Costs {
    double wake_loss_factor;  // Factor to model losses in annual energy production in a wind farm
                              // compared to the annual energy production at the turbine level (wakes
                              // mostly).
    double fixed_charge_rate;  // Fixed charge rate to compute the levelized cost of energy. See this
                               // for inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    double bos_per_kW;         // Balance of stations costs expressed in USD per kW. See this for
                               // inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    double opex_per_kW;        // Operational expenditures expressed in USD per kW. See this for
                               // inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    int turbine_number;  // Number of turbines in the park, used to compute levelized cost of energy.
                         // Often wind parks are assumed of 600 MW. See this for inspiration
                         // https://www.nrel.gov/docs/fy20osti/74598.pdf
    double labor_rate;   // Hourly loaded wage per worker including all benefits and overhead.  This
                         // is currently only applied to steel, column structures.
    double painting_rate;  // Cost per unit area for finishing and surface treatments.  This is
                           // currently only applied to steel, column structures.
    double blade_mass_cost_coeff;         // Regression-based blade cost/mass ratio
    double hub_mass_cost_coeff;           // Regression-based hub cost/mass ratio
    double pitch_system_mass_cost_coeff;  // Regression-based pitch system cost/mass ratio
    double spinner_mass_cost_coeff;       // Regression-based spinner cost/mass ratio
    double lss_mass_cost_coeff;           // Regression-based low speed shaft cost/mass ratio
    double bearing_mass_cost_coeff;       // Regression-based bearing cost/mass ratio
    double gearbox_mass_cost_coeff;       // Regression-based gearbox cost/mass ratio
    double hss_mass_cost_coeff;           // Regression-based high speed side cost/mass ratio
    double generator_mass_cost_coeff;     // Regression-based generator cost/mass ratio
    double bedplate_mass_cost_coeff;      // Regression-based bedplate cost/mass ratio
    double yaw_mass_cost_coeff;           // Regression-based yaw system cost/mass ratio
    double converter_mass_cost_coeff;     // Regression-based converter cost/mass ratio
    double transformer_mass_cost_coeff;   // Regression-based transformer cost/mass ratio
    double hvac_mass_cost_coeff;          // Regression-based HVAC system cost/mass ratio
    double cover_mass_cost_coeff;         // Regression-based nacelle cover cost/mass ratio
    double elec_connec_machine_rating_cost_coeff;  // Regression-based electrical plant connection
                                                   // cost/rating ratio
    double platforms_mass_cost_coeff;           // Regression-based nacelle platform cost/mass ratio
    double tower_mass_cost_coeff;               // Regression-based tower cost/mass ratio
    double controls_machine_rating_cost_coeff;  // Regression-based controller and sensor system
                                                // cost/rating ratio
    double crane_cost;                          // crane cost if present
    double electricity_price;  // Electricity price used to compute value in beyond lcoe metrics
    double
        reserve_margin_price;  // Reserve margin price used to compute value in beyond lcoe metrics
    double capacity_credit;    // Capacity credit used to compute value in beyond lcoe metrics
    double
        benchmark_price;  // Benchmark price used to nondimensionalize value in beyond lcoe metrics
};

// Turbine
struct Turbine {
    std::string comments;  // Description of the model
    std::string name;      // Name of the turbine
    Assembly assembly;
    Components components;
    std::vector<Airfoils> airfoils;    // Database of airfoils
    std::vector<Materials> materials;  // Database of the materials
    Control control;
    Environment environment;
    Bos bos;
    Costs costs;

    void parse(const YAML::Node& node) {
        comments = node["comments"] ? node["comments"].as<std::string>() : "";
        name = node["name"] ? node["name"].as<std::string>() : "";

        if (node["assembly"]) {
            assembly.parse(node["assembly"]);
        }

        for (std::size_t i = 0; i < node["materials"].size(); i++) {
            Materials mats;
            mats.parse(node["materials"][i]);
            materials.push_back(mats);
        }
    }
};
