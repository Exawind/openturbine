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
    int number_of_blades{};         // Number of blades of the rotor
    double rotor_diameter{};  // Diameter of the rotor, defined as two times the blade length plus
                              // the hub diameter
    double hub_height{};   // Height of the hub center over the ground (land-based) or the mean sea
                           // level (offshore)
    double rated_power{};  // Nameplate power of the turbine, i.e. the rated electrical output of the
                           // generator.
    double lifetime{};     // Turbine design lifetime in years.

    void parse(const YAML::Node& node) {
        turbine_class = node["turbine_class"].as<std::string>("");
        turbulence_class = node["turbulence_class"].as<std::string>("");
        drivetrain = node["drivetrain"].as<std::string>("");
        rotor_orientation = node["rotor_orientation"].as<std::string>("");
        number_of_blades = node["number_of_blades"].as<int>(0);
        rotor_diameter = node["rotor_diameter"].as<double>(0.);
        hub_height = node["hub_height"].as<double>(0.);
        rated_power = node["rated_power"].as<double>(0.);
        lifetime = node["lifetime"].as<double>(0.);
    }
};

// AirfoilPosition
struct AirfoilPosition {
    std::vector<double> grid;
    std::vector<std::string> labels;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        labels = node["labels"].as<std::vector<std::string>>(std::vector<std::string>{});
    }
};

// Chord
struct Chord {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Twist
struct Twist {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// PitchAxis
struct PitchAxis {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// TDividedByC
struct TDividedByC {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// LDividedByD
struct LDividedByD {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// CD
struct CD {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// StallMargin
struct StallMargin {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// X
struct X {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Y
struct Y {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Z
struct Z {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// The reference system is located at blade root, with z aligned with the pitch axis, x pointing
// towards the suction sides of the airfoils (standard prebend will be negative) and y pointing to
// the trailing edge (standard sweep will be positive)
struct ReferenceAxis {
    X x;
    Y y;
    Z z;

    void parse(const YAML::Node& node) {
        if (node["x"]) {
            x.parse(node["x"]);
        }
        if (node["y"]) {
            y.parse(node["y"]);
        }
        if (node["z"]) {
            z.parse(node["z"]);
        }
    }
};

// Rthick
struct Rthick {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// OuterShapeBem
struct OuterShapeBem {
    AirfoilPosition airfoil_position;
    Chord chord;
    Twist twist;
    PitchAxis pitch_axis;
    TDividedByC t_divided_by_c;
    LDividedByD l_divided_by_d;
    CD c_d;
    StallMargin stall_margin;
    ReferenceAxis reference_axis;
    Rthick rthick;

    void parse(const YAML::Node& node) {
        if (node["airfoil_position"]) {
            airfoil_position.parse(node["airfoil_position"]);
        }
        if (node["chord"]) {
            chord.parse(node["chord"]);
        }
        if (node["twist"]) {
            twist.parse(node["twist"]);
        }
        if (node["pitch_axis"]) {
            pitch_axis.parse(node["pitch_axis"]);
        }
        if (node["t_divided_by_c"]) {
            t_divided_by_c.parse(node["t_divided_by_c"]);
        }
        if (node["l_divided_by_d"]) {
            l_divided_by_d.parse(node["l_divided_by_d"]);
        }
        if (node["c_d"]) {
            c_d.parse(node["c_d"]);
        }
        if (node["stall_margin"]) {
            stall_margin.parse(node["stall_margin"]);
        }
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["rthick"]) {
            rthick.parse(node["rthick"]);
        }
    }
};

// A
struct A {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// E
struct E {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// G
struct G {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// IX
struct IX {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// IY
struct IY {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// K
struct K {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Dm
struct Dm {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// KX
struct KX {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// KY
struct KY {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Pitch
struct Pitch {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// RiX
struct RiX {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// RiY
struct RiY {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// XCg
struct XCg {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// XE
struct XE {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// XSh
struct XSh {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// YCg
struct YCg {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// YE
struct YE {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// YSh
struct YSh {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Timoschenko beam as in HAWC2
struct TimoschenkoHawc {
    ReferenceAxis reference_axis;
    A a;
    E e;
    G g;
    IX i_x;
    IY i_y;
    K k;
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

    void parse(const YAML::Node& node) {
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["a"]) {
            a.parse(node["a"]);
        }
        if (node["e"]) {
            e.parse(node["e"]);
        }
        if (node["g"]) {
            g.parse(node["g"]);
        }
        if (node["i_x"]) {
            i_x.parse(node["i_x"]);
        }
        if (node["i_y"]) {
            i_y.parse(node["i_y"]);
        }
        if (node["k"]) {
            k.parse(node["k"]);
        }
        if (node["dm"]) {
            dm.parse(node["dm"]);
        }
        if (node["k_x"]) {
            k_x.parse(node["k_x"]);
        }
        if (node["k_y"]) {
            k_y.parse(node["k_y"]);
        }
        if (node["pitch"]) {
            pitch.parse(node["pitch"]);
        }
        if (node["ri_x"]) {
            ri_x.parse(node["ri_x"]);
        }
        if (node["ri_y"]) {
            ri_y.parse(node["ri_y"]);
        }
        if (node["x_cg"]) {
            x_cg.parse(node["x_cg"]);
        }
        if (node["x_e"]) {
            x_e.parse(node["x_e"]);
        }
        if (node["x_sh"]) {
            x_sh.parse(node["x_sh"]);
        }
        if (node["y_cg"]) {
            y_cg.parse(node["y_cg"]);
        }
        if (node["y_e"]) {
            y_e.parse(node["y_e"]);
        }
        if (node["y_sh"]) {
            y_sh.parse(node["y_sh"]);
        }
    }
};

// T11
struct T11 {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// T22
struct T22 {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Ea
struct Ea {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// E11
struct E11 {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// E22
struct E22 {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Gj
struct Gj {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// XCe
struct XCe {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// YCe
struct YCe {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// DeltaTheta
struct DeltaTheta {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// J1
struct J1 {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// J2
struct J2 {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// J3
struct J3 {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Geometrically exact beams with simplified properties
struct CpLambdaBeam {
    ReferenceAxis reference_axis;
    T11 t11;
    T22 t22;
    Ea ea;
    E11 e11;
    E22 e22;
    Gj gj;
    XCe x_ce;
    YCe y_ce;
    Dm dm;
    DeltaTheta delta_theta;
    XSh x_sh;
    YSh y_sh;
    J1 j1;
    J2 j2;
    J3 j3;
    XCg x_cg;
    YCg y_cg;

    void parse(const YAML::Node& node) {
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["t11"]) {
            t11.parse(node["t11"]);
        }
        if (node["t22"]) {
            t22.parse(node["t22"]);
        }
        if (node["ea"]) {
            ea.parse(node["ea"]);
        }
        if (node["e11"]) {
            e11.parse(node["e11"]);
        }
        if (node["e22"]) {
            e22.parse(node["e22"]);
        }
        if (node["gj"]) {
            gj.parse(node["gj"]);
        }
        if (node["x_ce"]) {
            x_ce.parse(node["x_ce"]);
        }
        if (node["y_ce"]) {
            y_ce.parse(node["y_ce"]);
        }
        if (node["dm"]) {
            dm.parse(node["dm"]);
        }
        if (node["delta_theta"]) {
            delta_theta.parse(node["delta_theta"]);
        }
        if (node["x_sh"]) {
            x_sh.parse(node["x_sh"]);
        }
        if (node["y_sh"]) {
            y_sh.parse(node["y_sh"]);
        }
        if (node["j1"]) {
            j1.parse(node["j1"]);
        }
        if (node["j2"]) {
            j2.parse(node["j2"]);
        }
        if (node["j3"]) {
            j3.parse(node["j3"]);
        }
        if (node["x_cg"]) {
            x_cg.parse(node["x_cg"]);
        }
        if (node["y_cg"]) {
            y_cg.parse(node["y_cg"]);
        }
    }
};

// StiffMatrix
struct StiffMatrix {
    std::vector<double> grid;
    std::vector<std::vector<double>> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        if (node["values"]) {
            std::transform(
                node["values"].begin(), node["values"].end(), std::back_inserter(values),
                [](const auto& item) {
                    //  use 'template' keyword to treat 'as' as a dependent template name
                    return item.template as<std::vector<double>>();
                }
            );
        }
    }
};

// SixXSix
struct SixXSix {
    ReferenceAxis reference_axis;
    StiffMatrix stiff_matrix;

    void parse(const YAML::Node& node) {
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["stiff_matrix"]) {
            stiff_matrix.parse(node["stiff_matrix"]);
        }
    }
};

// ElasticPropertiesMb
struct ElasticPropertiesMb {
    TimoschenkoHawc timoschenko_hawc;
    CpLambdaBeam cp_lambda_beam;
    SixXSix six_x_six;

    void parse(const YAML::Node& node) {
        if (node["timoschenko_hawc"]) {
            timoschenko_hawc.parse(node["timoschenko_hawc"]);
        }
        if (node["cp_lambda_beam"]) {
            cp_lambda_beam.parse(node["cp_lambda_beam"]);
        }
        if (node["six_x_six"]) {
            six_x_six.parse(node["six_x_six"]);
        }
    }
};

// Root
struct Root {
    double d_f;        // Diameter of the fastener, default is M30, so 0.03 meters
    double sigma_max;  // Max stress on bolt

    void parse(const YAML::Node& node) {
        d_f = node["d_f"].as<double>(0.);
        sigma_max = node["sigma_max"].as<double>(0.);
    }
};

// non-dimensional location of the point along the non-dimensional arc length
struct StartNdArc {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed;  // Name of the layer to lock the edge

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
        fixed = node["fixed"].as<std::string>("");
    }
};

// non-dimensional location of the point along the non-dimensional arc length
struct EndNdArc {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed;  // Name of the layer to lock the edge

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
        fixed = node["fixed"].as<std::string>("");
    }
};

// rotation of the chord axis around the pitch axis
struct Rotation {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed;  // Name of the layer to lock the edge

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
        fixed = node["fixed"].as<std::string>("");
    }
};

// dimensional offset in respect to the pitch axis along the x axis, which is the chord line rotated
// by a user-defined angle. Negative values move the midpoint towards the leading edge, positive
// towards the trailing edge
struct OffsetYPa {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Webs
struct Webs {
    std::string name;  // structural component identifier
    StartNdArc start_nd_arc;
    EndNdArc end_nd_arc;
    Rotation rotation;
    OffsetYPa offset_y_pa;

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        if (node["start_nd_arc"]) {
            start_nd_arc.parse(node["start_nd_arc"]);
        }
        if (node["end_nd_arc"]) {
            end_nd_arc.parse(node["end_nd_arc"]);
        }
        if (node["rotation"]) {
            rotation.parse(node["rotation"]);
        }
        if (node["offset_y_pa"]) {
            offset_y_pa.parse(node["offset_y_pa"]);
        }
    }
};

// thickness of the laminate
struct Thickness {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// number of plies of the laminate
struct NPlies {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// orientation of the fibers
struct FiberOrientation {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// dimensional width of the component along the arc
struct Width {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// non-dimensional location of the point along the non-dimensional arc length
struct MidpointNdArc {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed;  // Name of the layer to lock the edge

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
        fixed = node["fixed"].as<std::string>("");
    }
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

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        material = node["material"].as<std::string>("");
        web = node["web"].as<std::string>("");
        if (node["thickness"]) {
            thickness.parse(node["thickness"]);
        }
        if (node["n_plies"]) {
            n_plies.parse(node["n_plies"]);
        }
        if (node["fiber_orientation"]) {
            fiber_orientation.parse(node["fiber_orientation"]);
        }
        if (node["width"]) {
            width.parse(node["width"]);
        }
        if (node["midpoint_nd_arc"]) {
            midpoint_nd_arc.parse(node["midpoint_nd_arc"]);
        }
        if (node["start_nd_arc"]) {
            start_nd_arc.parse(node["start_nd_arc"]);
        }
        if (node["end_nd_arc"]) {
            end_nd_arc.parse(node["end_nd_arc"]);
        }
        if (node["rotation"]) {
            rotation.parse(node["rotation"]);
        }
        if (node["offset_y_pa"]) {
            offset_y_pa.parse(node["offset_y_pa"]);
        }
    }
};

// This is a spanwise joint along the blade, usually adopted to ease transportation constraints.
// WISDEM currently supports a single joint.
struct Joint {
    double position{};          // Spanwise position of the segmentation joint.
    double mass{};              // Mass of the joint.
    double cost{};              // Cost of the joint.
    std::string bolt;           // Bolt size for the blade bolted joint
    double nonmaterial_cost{};  // Cost of the joint not from materials.
    std::string
        reinforcement_layer_ss;  // Layer identifier for the joint reinforcement on the suction side
    std::string
        reinforcement_layer_ps;  // Layer identifier for the joint reinforcement on the pressure side

    void parse(const YAML::Node& node) {
        position = node["position"].as<double>(0.);
        mass = node["mass"].as<double>(0.);
        cost = node["cost"].as<double>(0.);
        bolt = node["bolt"].as<std::string>("");
        nonmaterial_cost = node["nonmaterial_cost"].as<double>(0.);
        reinforcement_layer_ss = node["reinforcement_layer_ss"].as<std::string>("");
        reinforcement_layer_ps = node["reinforcement_layer_ps"].as<std::string>("");
    }
};

// InternalStructure2DFem
struct InternalStructure2DFem {
    Root root{};
    ReferenceAxis reference_axis;
    std::vector<Webs> webs;      // ...
    std::vector<Layers> layers;  // ...
    Joint joint;  // This is a spanwise joint along the blade, usually adopted to ease transportation
                  // constraints. WISDEM currently supports a single joint.

    void parse(const YAML::Node& node) {
        if (node["root"]) {
            root.parse(node["root"]);
        }
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["webs"]) {
            for (const auto& item : node["webs"]) {
                Webs x;
                x.parse(item);
                webs.push_back(x);
            }
        }
        if (node["layers"]) {
            for (const auto& item : node["layers"]) {
                Layers x;
                x.parse(item);
                layers.push_back(x);
            }
        }
        if (node["joint"]) {
            joint.parse(node["joint"]);
        }
    }
};

// Blade
struct Blade {
    OuterShapeBem outer_shape_bem;
    ElasticPropertiesMb elastic_properties_mb;
    InternalStructure2DFem internal_structure_2d_fem;

    void parse(const YAML::Node& node) {
        if (node["outer_shape_bem"]) {
            outer_shape_bem.parse(node["outer_shape_bem"]);
        }
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
        if (node["internal_structure_2d_fem"]) {
            internal_structure_2d_fem.parse(node["internal_structure_2d_fem"]);
        }
    }
};

// OuterShapeBem_1
struct OuterShapeBem_1 {
    double diameter;    // Diameter of the hub measured at the blade root positions.
    double cone_angle;  // Rotor precone angle, defined positive for both upwind and downwind rotors.
    double drag_coefficient;  // Equivalent drag coefficient to compute the aerodynamic forces
                              // generated on the hub.

    void parse(const YAML::Node& node) {
        diameter = node["diameter"].as<double>(0.);
        cone_angle = node["cone_angle"].as<double>(0.);
        drag_coefficient = node["drag_coefficient"].as<double>(0.);
    }
};

// ElasticPropertiesMb_1
struct ElasticPropertiesMb_1 {
    double system_mass{};  // Mass of the hub system, which includes the hub, the spinner, the blade
                           // bearings, the pitch actuators, the cabling, ....
    std::vector<double>
        system_inertia;  // Inertia of the hub system, on the hub reference system, which has the x
                         // aligned with the rotor axis, and y and z perpendicular to it.
    std::vector<double> system_center_mass;  // Center of mass of the hub system. Work in progress.

    void parse(const YAML::Node& node) {
        system_mass = node["system_mass"].as<double>(0.);
        system_inertia = node["system_inertia"].as<std::vector<double>>(std::vector<double>{});
        system_center_mass =
            node["system_center_mass"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Hub
struct Hub {
    OuterShapeBem_1 outer_shape_bem{};
    ElasticPropertiesMb_1 elastic_properties_mb;

    void parse(const YAML::Node& node) {
        if (node["outer_shape_bem"]) {
            outer_shape_bem.parse(node["outer_shape_bem"]);
        }
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
    }
};

// User input override of generator rpm-efficiency values, with rpm as grid input and eff as values
// input
struct GeneratorRpmEfficiencyUser {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Thickness of the hollow elliptical bedplate used in direct drive configurations
struct BedplateWallThickness {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Inputs to WISDEM specific drivetrain sizing tool, DrivetrainSE
struct Drivetrain {
    double uptilt{};            // Tilt angle of the nacelle, always defined positive.
    double distance_tt_hub{};   // Vertical distance between the tower top and the hub center.
    double distance_hub_mb{};   // Distance from hub flange to first main bearing along shaft.
    double distance_mb_mb{};    // Distance from first to second main bearing along shaft.
    double overhang{};          // Horizontal distance between the tower axis and the rotor apex.
    double generator_length{};  // Length of generator along the shaft
    double generator_radius_user{};  // User input override of generator radius, only used when using
                                     // simple generator scaling
    double generator_mass_user{};    // User input override of generator mass, only used when using
                                     // simple generator mass scaling
    GeneratorRpmEfficiencyUser
        generator_rpm_efficiency_user{};  // User input override of generator rpm-efficiency values,
                                          // with rpm as grid input and eff as values input
    double gear_ratio{};  // Gear ratio of the drivetrain. Set it to 1 for direct drive machines.
    double gearbox_length_user{};  // User input override of gearbox length along shaft, only used
                                   // when using gearbox_mass_user is > 0
    double gearbox_radius_user{};      // User input override of gearbox radius, only used when using
                                       // gearbox_mass_user is > 0
    double gearbox_mass_user{};        // User input override of gearbox mass
    double gearbox_efficiency{};       // Efficiency of the gearbox system.
    double damping_ratio{};            // Damping ratio for the drivetrain system
    std::vector<double> lss_diameter;  // Diameter of the low speed shaft at beginning
                                       // (generator/gearbox) and end (hub) points
    std::vector<double> lss_wall_thickness;  // Thickness of the low speed shaft at beginning
                                             // (generator/gearbox) and end (hub) points
    std::string lss_material;                // Material name identifier
    double hss_length{};                     // Length of the high speed shaft
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
    double bedplate_flange_width{};  // Bedplate I-beam flange width used in geared configurations
    double bedplate_flange_thickness{};  // Bedplate I-beam flange thickness used in geared
                                         // configurations
    double bedplate_web_thickness{};  // Bedplate I-beam web thickness used in geared configurations
    double brake_mass_user{};  // Override regular regression-based calculation of brake mass with
                               // this value
    double hvac_mass_coefficient{};  // Regression-based scaling coefficient on machine rating to get
                                     // HVAC system mass
    double converter_mass_user{};  // Override regular regression-based calculation of converter mass
                                   // with this value
    double transformer_mass_user{};  // Override regular regression-based calculation of transformer
                                     // mass with this value
    std::string bedplate_material;   // Material name identifier
    std::string mb1type;             // Type of bearing for first main bearing
    std::string mb2type;             // Type of bearing for second main bearing
    bool uptower{};  // If power electronics are located uptower (True) or at tower base (False)
    std::string gear_configuration;   // 3-letter string of Es or Ps to denote epicyclic or parallel
                                      // gear configuration
    std::vector<int> planet_numbers;  // Number of planets for epicyclic stages (use 0 for parallel)

    void parse(const YAML::Node& node) {
        uptilt = node["uptilt"].as<double>(0.);
        distance_tt_hub = node["distance_tt_hub"].as<double>(0.);
        distance_hub_mb = node["distance_hub_mb"].as<double>(0.);
        distance_mb_mb = node["distance_mb_mb"].as<double>(0.);
        overhang = node["overhang"].as<double>(0.);
        generator_length = node["generator_length"].as<double>(0.);
        generator_radius_user = node["generator_radius_user"].as<double>(0.);
        generator_mass_user = node["generator_mass_user"].as<double>(0.);
        if (node["generator_rpm_efficiency_user"]) {
            generator_rpm_efficiency_user.parse(node["generator_rpm_efficiency_user"]);
        }
        gear_ratio = node["gear_ratio"].as<double>(0.);
        gearbox_length_user = node["gearbox_length_user"].as<double>(0.);
        gearbox_radius_user = node["gearbox_radius_user"].as<double>(0.);
        gearbox_mass_user = node["gearbox_mass_user"].as<double>(0.);
        gearbox_efficiency = node["gearbox_efficiency"].as<double>(0.);
        damping_ratio = node["damping_ratio"].as<double>(0.);
        lss_diameter = node["lss_diameter"].as<std::vector<double>>(std::vector<double>{});
        lss_wall_thickness =
            node["lss_wall_thickness"].as<std::vector<double>>(std::vector<double>{});
        lss_material = node["lss_material"].as<std::string>("");
        hss_length = node["hss_length"].as<double>(0.);
        hss_diameter = node["hss_diameter"].as<std::vector<double>>(std::vector<double>{});
        hss_wall_thickness =
            node["hss_wall_thickness"].as<std::vector<double>>(std::vector<double>{});
        hss_material = node["hss_material"].as<std::string>("");
        nose_diameter = node["nose_diameter"].as<std::vector<double>>(std::vector<double>{});
        nose_wall_thickness =
            node["nose_wall_thickness"].as<std::vector<double>>(std::vector<double>{});
        if (node["bedplate_wall_thickness"]) {
            bedplate_wall_thickness.parse(node["bedplate_wall_thickness"]);
        }
        bedplate_flange_width = node["bedplate_flange_width"].as<double>(0.);
        bedplate_flange_thickness = node["bedplate_flange_thickness"].as<double>(0.);
        bedplate_web_thickness = node["bedplate_web_thickness"].as<double>(0.);
        brake_mass_user = node["brake_mass_user"].as<double>(0.);
        hvac_mass_coefficient = node["hvac_mass_coefficient"].as<double>(0.);
        converter_mass_user = node["converter_mass_user"].as<double>(0.);
        transformer_mass_user = node["transformer_mass_user"].as<double>(0.);
        bedplate_material = node["bedplate_material"].as<std::string>("");
        mb1type = node["mb1type"].as<std::string>("");
        mb2type = node["mb2type"].as<std::string>("");
        uptower = node["uptower"].as<bool>(false);
        gear_configuration = node["gear_configuration"].as<std::string>("");
        planet_numbers = node["planet_numbers"].as<std::vector<int>>(std::vector<int>{});
    }
};

// Generator
struct Generator {
    double mass_coefficient{};  // When not doing a detailed generator design, use a simplified
                                // approach to generator scaling. This input allows for overriding of
                                // the regression-based scaling coefficient to obtain generator mass
    std::string generator_type;
    double b_r{};          // Words
    double p_fe0e{};       // Words
    double p_fe0h{};       // Words
    double s_n{};          // Words
    double s_nmax{};       // Words
    double alpha_p{};      // Words
    double b_r_tau_r{};    // Words
    double b_ro{};         // Words
    double b_s_tau_s{};    // Words
    double b_so{};         // Words
    double cofi{};         // Words
    double freq{};         // Words
    double h_i{};          // Words
    double h_sy0{};        // Words
    double h_w{};          // Words
    double k_fes{};        // Words
    double k_fillr{};      // Words
    double k_fills{};      // Words
    double k_s{};          // Words
    int m{};               // Words
    double mu_0{};         // Permittivity of free space
    double mu_r{};         // Words
    double p{};            // Words
    double phi{};          // Words
    int q1{};              // Words
    int q3{};              // Words
    double ratio_mw2pp{};  // Words
    double resist_cu{};    // Resistivity of copper
    double sigma{};        // Maximum allowable shear stress
    double y_tau_p{};      // Words
    double y_tau_pr{};     // Words
    double i_0{};          // Words
    double d_r{};          // Words
    double h_m{};          // Words
    double h_0{};          // Words
    double h_s{};          // Words
    double len_s{};        // Words
    double n_r{};          // Words
    double rad_ag{};       // Words
    double t_wr{};         // Words
    double n_s{};          // Words
    double b_st{};         // Words
    double d_s{};          // Words
    double t_ws{};         // Words
    double rho_copper{};   // Copper density
    double rho_fe{};       // Structural steel density
    double rho_fes{};      // Electrical steel density
    double rho_pm{};       // Permanent magnet density
    double c_cu{};         // Copper cost
    double c_fe{};         // Structural steel cost
    double c_fes{};        // Electrical steel cost
    double c_pm{};         // Permanent magnet cost

    void parse(const YAML::Node& node) {
        mass_coefficient = node["mass_coefficient"].as<double>(0.);
        generator_type = node["generator_type"].as<std::string>("");
        b_r = node["b_r"].as<double>(0.);
        p_fe0e = node["p_fe0e"].as<double>(0.);
        p_fe0h = node["p_fe0h"].as<double>(0.);
        s_n = node["s_n"].as<double>(0.);
        s_nmax = node["s_nmax"].as<double>(0.);
        alpha_p = node["alpha_p"].as<double>(0.);
        b_r_tau_r = node["b_r_tau_r"].as<double>(0.);
        b_ro = node["b_ro"].as<double>(0.);
        b_s_tau_s = node["b_s_tau_s"].as<double>(0.);
        b_so = node["b_so"].as<double>(0.);
        cofi = node["cofi"].as<double>(0.);
        freq = node["freq"].as<double>(0.);
        h_i = node["h_i"].as<double>(0.);
        h_sy0 = node["h_sy0"].as<double>(0.);
        h_w = node["h_w"].as<double>(0.);
        k_fes = node["k_fes"].as<double>(0.);
        k_fillr = node["k_fillr"].as<double>(0.);
        k_fills = node["k_fills"].as<double>(0.);
        k_s = node["k_s"].as<double>(0.);
        m = node["m"].as<int>(0);
        mu_0 = node["mu_0"].as<double>(0.);
        mu_r = node["mu_r"].as<double>(0.);
        p = node["p"].as<double>(0.);
        phi = node["phi"].as<double>(0.);
        q1 = node["q1"].as<int>(0);
        q3 = node["q3"].as<int>(0);
        ratio_mw2pp = node["ratio_mw2pp"].as<double>(0.);
        resist_cu = node["resist_cu"].as<double>(0.);
        sigma = node["sigma"].as<double>(0.);
        y_tau_p = node["y_tau_p"].as<double>(0.);
        y_tau_pr = node["y_tau_pr"].as<double>(0.);
        i_0 = node["i_0"].as<double>(0.);
        d_r = node["d_r"].as<double>(0.);
        h_m = node["h_m"].as<double>(0.);
        h_0 = node["h_0"].as<double>(0.);
        h_s = node["h_s"].as<double>(0.);
        len_s = node["len_s"].as<double>(0.);
        n_r = node["n_r"].as<double>(0.);
        rad_ag = node["rad_ag"].as<double>(0.);
        t_wr = node["t_wr"].as<double>(0.);
        n_s = node["n_s"].as<double>(0.);
        b_st = node["b_st"].as<double>(0.);
        d_s = node["d_s"].as<double>(0.);
        t_ws = node["t_ws"].as<double>(0.);
        rho_copper = node["rho_copper"].as<double>(0.);
        rho_fe = node["rho_fe"].as<double>(0.);
        rho_fes = node["rho_fes"].as<double>(0.);
        rho_pm = node["rho_pm"].as<double>(0.);
        c_cu = node["c_cu"].as<double>(0.);
        c_fe = node["c_fe"].as<double>(0.);
        c_fes = node["c_fes"].as<double>(0.);
        c_pm = node["c_pm"].as<double>(0.);
    }
};

// Nacelle
struct Nacelle {
    Drivetrain drivetrain;  // Inputs to WISDEM specific drivetrain sizing tool, DrivetrainSE
    Generator generator;

    void parse(const YAML::Node& node) {
        if (node["drivetrain"]) {
            drivetrain.parse(node["drivetrain"]);
        }
        if (node["generator"]) {
            generator.parse(node["generator"]);
        }
    }
};

// OuterDiameter
struct OuterDiameter {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// DragCoefficient
struct DragCoefficient {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// OuterShapeBem_2
struct OuterShapeBem_2 {
    ReferenceAxis reference_axis;
    OuterDiameter outer_diameter;
    DragCoefficient drag_coefficient;

    void parse(const YAML::Node& node) {
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["outer_diameter"]) {
            outer_diameter.parse(node["outer_diameter"]);
        }
        if (node["drag_coefficient"]) {
            drag_coefficient.parse(node["drag_coefficient"]);
        }
    }
};

// InternalStructure2DFem_1
struct InternalStructure2DFem_1 {
    double outfitting_factor{};  // Scaling factor for the tower mass to account for auxiliary
                                 // structures, such as elevator, ladders, cables, platforms, etc
    ReferenceAxis reference_axis;
    std::vector<Layers> layers;  // ...

    void parse(const YAML::Node& node) {
        outfitting_factor = node["outfitting_factor"].as<double>(0.);
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["layers"]) {
            for (const auto& item : node["layers"]) {
                Layers x;
                x.parse(item);
                layers.push_back(x);
            }
        }
    }
};

// Tower
struct Tower {
    OuterShapeBem_2 outer_shape_bem;
    ElasticPropertiesMb elastic_properties_mb;
    InternalStructure2DFem_1 internal_structure_2d_fem;

    void parse(const YAML::Node& node) {
        if (node["outer_shape_bem"]) {
            outer_shape_bem.parse(node["outer_shape_bem"]);
        }
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
        if (node["internal_structure_2d_fem"]) {
            internal_structure_2d_fem.parse(node["internal_structure_2d_fem"]);
        }
    }
};

// OuterShape
struct OuterShape {
    ReferenceAxis reference_axis;
    OuterDiameter outer_diameter;
    DragCoefficient drag_coefficient;

    void parse(const YAML::Node& node) {
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["outer_diameter"]) {
            outer_diameter.parse(node["outer_diameter"]);
        }
        if (node["drag_coefficient"]) {
            drag_coefficient.parse(node["drag_coefficient"]);
        }
    }
};

// InternalStructure2DFem_2
struct InternalStructure2DFem_2 {
    double outfitting_factor{};  // Scaling factor for the tower mass to account for auxiliary
                                 // structures, such as elevator, ladders, cables, platforms, etc
    ReferenceAxis reference_axis;
    std::vector<Layers> layers;  // ...

    void parse(const YAML::Node& node) {
        outfitting_factor = node["outfitting_factor"].as<double>(0.);
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["layers"]) {
            for (const auto& item : node["layers"]) {
                Layers x;
                x.parse(item);
                layers.push_back(x);
            }
        }
    }
};

// Monopile
struct Monopile {
    double transition_piece_mass{};    // Total mass of transition piece
    double transition_piece_cost{};    // Total cost of transition piece
    double gravity_foundation_mass{};  // Total mass of gravity foundation addition onto monopile
    OuterShape outer_shape;
    ElasticPropertiesMb elastic_properties_mb;
    InternalStructure2DFem_2 internal_structure_2d_fem;

    void parse(const YAML::Node& node) {
        transition_piece_mass = node["transition_piece_mass"].as<double>(0.);
        transition_piece_cost = node["transition_piece_cost"].as<double>(0.);
        gravity_foundation_mass = node["gravity_foundation_mass"].as<double>(0.);
        if (node["outer_shape"]) {
            outer_shape.parse(node["outer_shape"]);
        }
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
        if (node["internal_structure_2d_fem"]) {
            internal_structure_2d_fem.parse(node["internal_structure_2d_fem"]);
        }
    }
};

// Jacket
struct Jacket {
    double transition_piece_mass{};    // Total mass of transition piece
    double transition_piece_cost{};    // Total cost of transition piece
    double gravity_foundation_mass{};  // Total mass of gravity foundation addition onto monopile
    std::string material;              // Material of jacket members
    int n_bays{};            // Number of bays (x-joints) in the vertical direction for jackets.
    int n_legs{};            // Number of legs for jacket.
    double r_foot{};         // Radius of foot (bottom) of jacket, in meters.
    double r_head{};         // Radius of head (top) of jacket, in meters.
    double height{};         // Overall jacket height, meters.
    double leg_thickness{};  // Leg thickness, meters. Constant throughout each leg.
    std::vector<double> brace_diameters;
    std::vector<double> brace_thicknesses;
    std::vector<double> bay_spacing;
    std::vector<double> leg_spacing;
    bool x_mb{};            // Mud brace included if true.
    double leg_diameter{};  // Leg diameter, meters. Constant throughout each leg.

    void parse(const YAML::Node& node) {
        transition_piece_mass = node["transition_piece_mass"].as<double>(0.);
        transition_piece_cost = node["transition_piece_cost"].as<double>(0.);
        gravity_foundation_mass = node["gravity_foundation_mass"].as<double>(0.);
        material = node["material"].as<std::string>("");
        n_bays = node["n_bays"].as<int>(0);
        n_legs = node["n_legs"].as<int>(0);
        r_foot = node["r_foot"].as<double>(0.);
        r_head = node["r_head"].as<double>(0.);
        height = node["height"].as<double>(0.);
        leg_thickness = node["leg_thickness"].as<double>(0.);
        brace_diameters = node["brace_diameters"].as<std::vector<double>>(std::vector<double>{});
        brace_thicknesses = node["brace_thicknesses"].as<std::vector<double>>(std::vector<double>{});
        bay_spacing = node["bay_spacing"].as<std::vector<double>>(std::vector<double>{});
        leg_spacing = node["leg_spacing"].as<std::vector<double>>(std::vector<double>{});
        x_mb = node["x_mb"].as<bool>(false);
        leg_diameter = node["leg_diameter"].as<double>(0.);
    }
};

// If this joint is compliant is certain DOFs, then specify which are compliant (True) in the
// member/element coordinate system).  If not specified, default is all entries are False (completely
// rigid).  For instance, a ball joint would be Rx=Ry=Rz=False, Rxx=Ryy=Rzz=True
struct Reactions {
    bool rx{};
    bool ry{};
    bool rz{};
    bool rxx{};
    bool ryy{};
    bool rzz{};
    std::vector<double> euler;  // Euler angles [alpha, beta, gamma] that describe the rotation of
                                // the Reaction coordinate system relative to the global coordinate
                                // system α is a rotation around the z axis, β is a rotation around
                                // the x' axis, γ is a rotation around the z" axis.

    void parse(const YAML::Node& node) {
        rx = node["rx"].as<bool>(false);
        ry = node["ry"].as<bool>(false);
        rz = node["rz"].as<bool>(false);
        rxx = node["rxx"].as<bool>(false);
        ryy = node["ryy"].as<bool>(false);
        rzz = node["rzz"].as<bool>(false);
        euler = node["euler"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Joints
struct Joints {
    std::string name;  // Unique name of the joint (node)
    std::vector<double>
        location;  // Coordinates (x,y,z or r,θ,z) of the joint in the global coordinate system.
    bool transition{};   // Whether the transition piece and turbine tower attach at this node
    bool cylindrical{};  // Whether to use cylindrical coordinates (r,θ,z), with (r,θ) lying in the
                         // x/y-plane, instead of Cartesian coordinates.
    Reactions reactions;  // If this joint is compliant is certain DOFs, then specify which are
                          // compliant (True) in the member/element coordinate system).  If not
                          // specified, default is all entries are False (completely rigid).  For
                          // instance, a ball joint would be Rx=Ry=Rz=False, Rxx=Ryy=Rzz=True

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        location = node["location"].as<std::vector<double>>(std::vector<double>{});
        transition = node["transition"].as<bool>(false);
        cylindrical = node["cylindrical"].as<bool>(false);
        if (node["reactions"]) {
            reactions.parse(node["reactions"]);
        }
    }
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
    double
        rotation{};  // Angle between principle axes of the cross-section and the member coordinate
                     // system.  Essentially the rotation of the member if both joints were placed
                     // on the global x-y axis with the first side length along the z-axis

    void parse(const YAML::Node& node) {
        shape = node["shape"].as<std::string>("");
        if (node["outer_diameter"]) {
            outer_diameter.parse(node["outer_diameter"]);
        }
        side_lengths1 = node["side_lengths1"].as<std::vector<double>>(std::vector<double>{});
        side_lengths2 = node["side_lengths2"].as<std::vector<double>>(std::vector<double>{});
        angles = node["angles"].as<std::vector<double>>(std::vector<double>{});
        rotation = node["rotation"].as<double>(0.);
    }
};

// Layers_1
struct Layers_1 {
    std::string name;      // structural component identifier
    std::string material;  // material identifier
    Thickness thickness;   // Gridded values describing thickness along non-dimensional axis from
                           // joint1 to joint2

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        material = node["material"].as<std::string>("");
        if (node["thickness"]) {
            thickness.parse(node["thickness"]);
        }
    }
};

// RingStiffeners
struct RingStiffeners {
    std::string material;  // material identifier
    double flange_thickness{};
    double flange_width{};
    double web_height{};
    double web_thickness{};
    double spacing{};  // Spacing between stiffeners in non-dimensional grid coordinates. Value of
                       // 0.0 means no stiffeners

    void parse(const YAML::Node& node) {
        material = node["material"].as<std::string>("");
        flange_thickness = node["flange_thickness"].as<double>(0.);
        flange_width = node["flange_width"].as<double>(0.);
        web_height = node["web_height"].as<double>(0.);
        web_thickness = node["web_thickness"].as<double>(0.);
        spacing = node["spacing"].as<double>(0.);
    }
};

// LongitudinalStiffeners
struct LongitudinalStiffeners {
    std::string material;  // material identifier
    double flange_thickness{};
    double flange_width{};
    double web_height{};
    double web_thickness{};
    double spacing{};  // Spacing between stiffeners in angle (radians). Value of 0.0 means no
                       // stiffeners

    void parse(const YAML::Node& node) {
        material = node["material"].as<std::string>("");
        flange_thickness = node["flange_thickness"].as<double>(0.);
        flange_width = node["flange_width"].as<double>(0.);
        web_height = node["web_height"].as<double>(0.);
        web_thickness = node["web_thickness"].as<double>(0.);
        spacing = node["spacing"].as<double>(0.);
    }
};

// Bulkhead
struct Bulkhead {
    std::string material;  // material identifier
    Thickness
        thickness;  // thickness of the bulkhead at non-dimensional locations of the member [0..1]

    void parse(const YAML::Node& node) {
        material = node["material"].as<std::string>("");
        if (node["thickness"]) {
            thickness.parse(node["thickness"]);
        }
    }
};

// Ballast
struct Ballast {
    bool variable_flag{};  // If true, then this ballast is variable and adjusted by control system.
                           // If false, then considered permanent
    std::string material;  // material identifier
    std::vector<double> grid;
    double volume{};  // Total volume of ballast (permanent ballast only)

    void parse(const YAML::Node& node) {
        variable_flag = node["variable_flag"].as<bool>(false);
        material = node["material"].as<std::string>("");
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        volume = node["volume"].as<double>(0.);
    }
};

// InternalStructure
struct InternalStructure {
    double outfitting_factor{};    // Scaling factor for the member mass to account for auxiliary
                                   // structures, such as elevator, ladders, cables, platforms,
                                   // fasteners, etc
    std::vector<Layers_1> layers;  // Material layer properties
    RingStiffeners ring_stiffeners;
    LongitudinalStiffeners longitudinal_stiffeners;
    Bulkhead bulkhead;
    std::vector<Ballast> ballast;  // Different types of permanent and/or variable ballast

    void parse(const YAML::Node& node) {
        outfitting_factor = node["outfitting_factor"].as<double>(0.);
        if (node["layers"]) {
            for (const auto& item : node["layers"]) {
                Layers_1 x;
                x.parse(item);
                layers.push_back(x);
            }
        }
        if (node["ring_stiffeners"]) {
            ring_stiffeners.parse(node["ring_stiffeners"]);
        }
        if (node["longitudinal_stiffeners"]) {
            longitudinal_stiffeners.parse(node["longitudinal_stiffeners"]);
        }
        if (node["bulkhead"]) {
            bulkhead.parse(node["bulkhead"]);
        }
        if (node["ballast"]) {
            for (const auto& item : node["ballast"]) {
                Ballast x;
                x.parse(item);
                ballast.push_back(x);
            }
        }
    }
};

// AxialJoints
struct AxialJoints {
    std::string name;  // Unique name of joint
    double grid{};     // Non-dimensional value along member axis

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        grid = node["grid"].as<double>(0.);
    }
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
    double ca{};       // User-defined added mass coefficient
    double cp{};       // User-defined pressure coefficient
    double cd{};       // User-defined drag coefficient

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        joint1 = node["joint1"].as<std::string>("");
        joint2 = node["joint2"].as<std::string>("");
        if (node["outer_shape"]) {
            outer_shape.parse(node["outer_shape"]);
        }
        if (node["internal_structure"]) {
            internal_structure.parse(node["internal_structure"]);
        }
        if (node["axial_joints"]) {
            for (const auto& item : node["axial_joints"]) {
                AxialJoints x;
                x.parse(item);
                axial_joints.push_back(x);
            }
        }
        ca = node["ca"].as<double>(0.);
        cp = node["cp"].as<double>(0.);
        cd = node["cd"].as<double>(0.);
    }
};

// RigidBodies
struct RigidBodies {
    std::string joint1;  // Name of joint/node connection
    double mass{};       // Mass of this rigid body
    double cost{};       // Cost of this rigid body
    std::vector<double>
        cm_offset;  // Offset from joint location to center of mass (CM) of body in dx, dy, dz
    std::vector<double> moments_of_inertia;  // Moments of inertia around body CM in Ixx, Iyy, Izz
    double ca{};                             // User-defined added mass coefficient
    double cp{};                             // User-defined pressure coefficient
    double cd{};                             // User-defined drag coefficient

    void parse(const YAML::Node& node) {
        joint1 = node["joint1"].as<std::string>("");
        mass = node["mass"].as<double>(0.);
        cost = node["cost"].as<double>(0.);
        cm_offset = node["cm_offset"].as<std::vector<double>>(std::vector<double>{});
        moments_of_inertia =
            node["moments_of_inertia"].as<std::vector<double>>(std::vector<double>{});
        ca = node["ca"].as<double>(0.);
        cp = node["cp"].as<double>(0.);
        cd = node["cd"].as<double>(0.);
    }
};

// Ontology definition for floating platforms (substructures) suitable for use with the WEIS
// co-design analysis tool
struct FloatingPlatform {
    std::vector<Joints> joints;
    std::vector<Members> members;
    std::vector<RigidBodies> rigid_bodies;
    double transition_piece_mass{};  // Total mass of transition piece
    double transition_piece_cost{};  // Total cost of transition piece

    void parse(const YAML::Node& node) {
        if (node["joints"]) {
            for (const auto& item : node["joints"]) {
                Joints x;
                x.parse(item);
                joints.push_back(x);
            }
        }
        if (node["members"]) {
            for (const auto& item : node["members"]) {
                Members x;
                x.parse(item);
                members.push_back(x);
            }
        }
        if (node["rigid_bodies"]) {
            for (const auto& item : node["rigid_bodies"]) {
                RigidBodies x;
                x.parse(item);
                rigid_bodies.push_back(x);
            }
        }
        transition_piece_mass = node["transition_piece_mass"].as<double>(0.);
        transition_piece_cost = node["transition_piece_cost"].as<double>(0.);
    }
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
    double node_mass{};    // Clump weight mass
    double node_volume{};  // Floater volume
    double drag_area{};    // Product of drag coefficient and projected area (assumed constant in all
                           // directions) to calculate a drag force for the node
    double added_mass{};   // Added mass coefficient used along with node volume to calculate added
                          // mass on node

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        node_type = node["node_type"].as<std::string>("");
        location = node["location"].as<std::vector<double>>(std::vector<double>{});
        joint = node["joint"].as<std::string>("");
        anchor_type = node["anchor_type"].as<std::string>("");
        fairlead_type = node["fairlead_type"].as<std::string>("");
        node_mass = node["node_mass"].as<double>(0.);
        node_volume = node["node_volume"].as<double>(0.);
        drag_area = node["drag_area"].as<double>(0.);
        added_mass = node["added_mass"].as<double>(0.);
    }
};

// Lines
struct Lines {
    std::string name;             // ID of this line
    std::string line_type;        // Reference to line type database
    double unstretched_length{};  // length of line segment prior to tensioning
    std::string node1;            // node id of first line connection
    std::string node2;            // node id of second line connection

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        line_type = node["line_type"].as<std::string>("");
        unstretched_length = node["unstretched_length"].as<double>(0.);
        node1 = node["node1"].as<std::string>("");
        node2 = node["node2"].as<std::string>("");
    }
};

// LineTypes
struct LineTypes {
    std::string name;   // Name of material or line type to be referenced by line segments
    double diameter{};  // the volume-equivalent diameter of the line – the diameter of a cylinder
                        // having the same displacement per unit length
    std::string type;   // Type of material for property lookup
    double mass_density{};  // mass per unit length (in air)
    double
        stiffness{};  // axial line stiffness, product of elasticity modulus and cross-sectional area
    double cost{};    // cost per unit length
    double breaking_load{};          // line break tension
    double damping{};                // internal damping (BA)
    double transverse_added_mass{};  // transverse added mass coefficient (with respect to line
                                     // displacement)
    double tangential_added_mass{};  // tangential added mass coefficient (with respect to line
                                     // displacement)
    double transverse_drag{};  // transverse drag coefficient (with respect to frontal area, d*l)
    double tangential_drag{};  // tangential drag coefficient (with respect to surface area, π*d*l)

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        diameter = node["diameter"].as<double>(0.);
        type = node["type"].as<std::string>("");
        mass_density = node["mass_density"].as<double>(0.);
        stiffness = node["stiffness"].as<double>(0.);
        cost = node["cost"].as<double>(0.);
        breaking_load = node["breaking_load"].as<double>(0.);
        damping = node["damping"].as<double>(0.);
        transverse_added_mass = node["transverse_added_mass"].as<double>(0.);
        tangential_added_mass = node["tangential_added_mass"].as<double>(0.);
        transverse_drag = node["transverse_drag"].as<double>(0.);
        tangential_drag = node["tangential_drag"].as<double>(0.);
    }
};

// AnchorTypes
struct AnchorTypes {
    std::string name;           // Name of anchor to be referenced by anchor_id in Nodes section
    std::string type;           // Type of anchor for property lookup
    double mass{};              // mass of the anchor
    double cost{};              // cost of the anchor
    double max_lateral_load{};  // Maximum lateral load (parallel to the sea floor) that the anchor
                                // can support
    double max_vertical_load{};  // Maximum vertical load (perpendicular to the sea floor) that the
                                 // anchor can support

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        type = node["type"].as<std::string>("");
        mass = node["mass"].as<double>(0.);
        cost = node["cost"].as<double>(0.);
        max_lateral_load = node["max_lateral_load"].as<double>(0.);
        max_vertical_load = node["max_vertical_load"].as<double>(0.);
    }
};

// Ontology definition for mooring systems suitable for use with the WEIS co-design analysis tool
struct Mooring {
    std::vector<Nodes> nodes;           // List of nodes in the mooring system
    std::vector<Lines> lines;           // List of all mooring line properties in the mooring system
    std::vector<LineTypes> line_types;  // List of mooring line properties used in the system
    std::vector<AnchorTypes> anchor_types;  // List of anchor properties used in the system

    void parse(const YAML::Node& node) {
        if (node["nodes"]) {
            for (const auto& item : node["nodes"]) {
                Nodes x;
                x.parse(item);
                nodes.push_back(x);
            }
        }
        if (node["lines"]) {
            for (const auto& item : node["lines"]) {
                Lines x;
                x.parse(item);
                lines.push_back(x);
            }
        }
        if (node["line_types"]) {
            for (const auto& item : node["line_types"]) {
                LineTypes x;
                x.parse(item);
                line_types.push_back(x);
            }
        }
        if (node["anchor_types"]) {
            for (const auto& item : node["anchor_types"]) {
                AnchorTypes x;
                x.parse(item);
                anchor_types.push_back(x);
            }
        }
    }
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

    void parse(const YAML::Node& node) {
        if (node["blade"]) {
            blade.parse(node["blade"]);
        }
        if (node["hub"]) {
            hub.parse(node["hub"]);
        }
        if (node["nacelle"]) {
            nacelle.parse(node["nacelle"]);
        }
        if (node["tower"]) {
            tower.parse(node["tower"]);
        }
        if (node["monopile"]) {
            monopile.parse(node["monopile"]);
        }
        if (node["jacket"]) {
            jacket.parse(node["jacket"]);
        }
        if (node["floating_platform"]) {
            floating_platform.parse(node["floating_platform"]);
        }
        if (node["mooring"]) {
            mooring.parse(node["mooring"]);
        }
    }
};

// Airfoil coordinates described from trailing edge (x=1) along the suction side (y>0) to leading
// edge (x=0) back to trailing edge (x=1) along the pressure side (y<0)
struct Coordinates {
    std::vector<double> x;
    std::vector<double> y;

    void parse(const YAML::Node& node) {
        x = node["x"].as<std::vector<double>>(std::vector<double>{});
        y = node["y"].as<std::vector<double>>(std::vector<double>{});
    }
};

// CL
struct CL {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// CM
struct CM {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"].as<std::vector<double>>(std::vector<double>{});
        values = node["values"].as<std::vector<double>>(std::vector<double>{});
    }
};

// Lift, drag and moment coefficients expressed in terms of angles of attack
struct Polars {
    std::string configuration;  // Text to identify the setup for the definition of the polars
    double re{};                // Reynolds number of the polars
    CL c_l;
    CD c_d;
    CM c_m;

    void parse(const YAML::Node& node) {
        configuration = node["configuration"].as<std::string>("");
        re = node["re"].as<double>(0.);
        if (node["c_l"]) {
            c_l.parse(node["c_l"]);
        }
        if (node["c_d"]) {
            c_d.parse(node["c_d"]);
        }
        if (node["c_m"]) {
            c_m.parse(node["c_m"]);
        }
    }
};

// Airfoils
struct Airfoils {
    std::string name;         // Name of the airfoil
    Coordinates coordinates;  // Airfoil coordinates described from trailing edge (x=1) along the
                              // suction side (y>0) to leading edge (x=0) back to trailing edge (x=1)
                              // along the pressure side (y<0)
    double relative_thickness{};  // Thickness of the airfoil expressed non-dimensional
    double aerodynamic_center{};  // Non-dimensional chordwise coordinate of the aerodynamic center
    std::vector<Polars> polars;   // Different sets of polars at varying conditions

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        if (node["coordinates"]) {
            coordinates.parse(node["coordinates"]);
        }
        relative_thickness = node["relative_thickness"].as<double>(0.);
        aerodynamic_center = node["aerodynamic_center"].as<double>(0.);
        if (node["polars"]) {
            for (const auto& item : node["polars"]) {
                Polars x;
                x.parse(item);
                polars.push_back(x);
            }
        }
    }
};

// Materials
struct Materials {
    std::string name;         // Name of the material
    std::string description;  // Optional field describing the material
    std::string source;       // Optional field describing where the data come from
    int orth{};               // Flag to switch between isotropic (0) and orthotropic (1) materials
    double rho{};  // Density of the material. For composites, this is the density of the laminate
                   // once cured
    std::variant<double, std::vector<double>> e;  // Stiffness modulus. For orthotropic materials, it
                                                  // consists of an array with E11, E22 and E33.
    std::variant<double, std::vector<double>>
        g;  // Shear stiffness modulus. For orthotropic materials, it consists of an array with G12,
            // G13 and G23
    std::variant<double, std::vector<double>>
        nu;  // Poisson ratio. For orthotropic materials, it consists of an array with nu12, nu13 and
             // nu23. For isotropic materials, a minimum of -1 and a maximum of 0.5 are imposed. No
             // limits are imposed to anisotropic materials.
    std::variant<double, std::vector<double>> alpha;  // Thermal coefficient of expansion
    std::variant<double, std::vector<double>>
        xt;  // Ultimate tensile strength. For orthotropic materials, it consists of an array with
             // the strength in directions 11, 22 and 33. The values must be positive
    std::variant<double, std::vector<double>>
        xc;  // Ultimate compressive strength. For orthotropic materials, it consists of an array
             // with the strength in directions 11, 22 and 33. The values must be positive
    std::variant<double, std::vector<double>>
        xy;  // Ultimate yield strength for metals. For orthotropic materials, it consists of an
             // array with the strength in directions 12, 13 and 23
    std::variant<double, std::vector<double>>
        s;  // Ultimate shear strength. For orthotropic materials, it consists of an array with the
            // strength in directions 12, 13 and 23
    double ply_t{};      // Ply thickness of the composite material
    double unit_cost{};  // Unit cost of the material. For composites, this is the unit cost of the
                         // dry fabric.
    double fvf{};               // Fiber volume fraction of the composite material
    double fwf{};               // Fiber weight fraction of the composite material
    double fiber_density{};     // Density of the fibers of a composite material.
    double area_density_dry{};  // Aerial density of a fabric of a composite material.
    int component_id{};         // Flag used by the NREL blade cost model
                         // https://www.nrel.gov/docs/fy19osti/73585.pdf to define the manufacturing
                         // process behind the laminate. 0 - coating, 1 - sandwich filler , 2 - shell
                         // skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
    double
        waste{};  // Fraction of material that ends up wasted during manufacturing. This quantity is
                  // used in the NREL blade cost model https://www.nrel.gov/docs/fy19osti/73585.pdf
    double roll_mass{};  // Mass of a fabric roll. This quantity is used in the NREL blade cost model
                         // https://www.nrel.gov/docs/fy19osti/73585.pdf
    double gic{};   // Mode 1 critical energy-release rate. It is used by NuMAD from Sandia National
                    // Laboratories
    double giic{};  // Mode 2 critical energy-release rate. It is used by NuMAD from Sandia National
                    // Laboratories
    double alp0{};  // Fracture angle under pure transverse compression. It is used by NuMAD from
                    // Sandia National Laboratories
    std::variant<double, std::vector<double>> a;  // Fatigue S/N curve fitting parameter S=A*N^(-1/m)
    std::variant<double, std::vector<double>> m;  // Fatigue S/N curve fitting parameter S=A*N^(-1/m)
    std::variant<double, std::vector<double>> r;  // Fatigue stress ratio

    void parse(const YAML::Node& node) {
        name = node["name"].as<std::string>("");
        description = node["description"].as<std::string>("");
        source = node["source"].as<std::string>("");
        orth = node["orth"].as<int>(0);
        rho = node["rho"].as<double>(0.);
        e = node["e"].as<double>(0.);
        g = node["g"].as<double>(0.);
        if (orth == 0) {
            nu = node["nu"].as<double>(0.);
        } else {
            nu = node["nu"].as<std::vector<double>>(std::vector<double>{});
        }
        if (orth == 0) {
            alpha = node["alpha"].as<double>(0.);
        } else {
            alpha = node["alpha"].as<std::vector<double>>(std::vector<double>{});
        }
        xt = node["xt"].as<double>(0.);
        xc = node["xc"].as<double>(0.);
        xy = node["xy"].as<double>(0.);
        s = node["s"].as<double>(0.);
        ply_t = node["ply_t"].as<double>(0.);
        unit_cost = node["unit_cost"].as<double>(0.);
        fvf = node["fvf"].as<double>(0.);
        fwf = node["fwf"].as<double>(0.);
        fiber_density = node["fiber_density"].as<double>(0.);
        area_density_dry = node["area_density_dry"].as<double>(0.);
        component_id = node["component_id"].as<int>(0);
        waste = node["waste"].as<double>(0.);
        roll_mass = node["roll_mass"].as<double>(0.);
        gic = node["gic"].as<double>(0.);
        giic = node["giic"].as<double>(0.);
        alp0 = node["alp0"].as<double>(0.);
        a = node["a"].as<double>(0.);
        m = node["m"].as<double>(0.);
        r = node["r"].as<double>(0.);
    }
};

// Supervisory
struct Supervisory {
    double vin{};    // Cut-in wind speed of the wind turbine.
    double vout{};   // Cut-out wind speed of the wind turbine.
    double maxts{};  // Maximum allowable blade tip speed.

    void parse(const YAML::Node& node) {
        vin = node["vin"].as<double>(0.);
        vout = node["vout"].as<double>(0.);
        maxts = node["maxts"].as<double>(0.);
    }
};

// Pitch_1
struct Pitch_1 {
    double min_pitch{};  // Minimum pitch angle, where the default is 0 degrees. It is used by the
                         // ROSCO controller (https://github.com/NREL/ROSCO)
    double max_pitch_rate{};  // Maximum pitch rate of the rotor blades.

    void parse(const YAML::Node& node) {
        min_pitch = node["min_pitch"].as<double>(0.);
        max_pitch_rate = node["max_pitch_rate"].as<double>(0.);
    }
};

// Torque
struct Torque {
    double max_torque_rate{};  // Maximum torque rate of the wind turbine generator.
    double tsr{};  // Rated tip speed ratio of the wind turbine. As default, it is maintained
                   // constant in region II.
    double vs_minspd{};  // Minimum rotor speed. It is used by the ROSCO controller
                         // (https://github.com/NREL/ROSCO)
    double vs_maxspd{};  // Maximum rotor speed. It is used by the ROSCO controller
                         // (https://github.com/NREL/ROSCO)

    void parse(const YAML::Node& node) {
        max_torque_rate = node["max_torque_rate"].as<double>(0.);
        tsr = node["tsr"].as<double>(0.);
        vs_minspd = node["vs_minspd"].as<double>(0.);
        vs_maxspd = node["vs_maxspd"].as<double>(0.);
    }
};

// Control
struct Control {
    Supervisory supervisory;
    Pitch_1 pitch;
    Torque torque;

    void parse(const YAML::Node& node) {
        if (node["supervisory"]) {
            supervisory.parse(node["supervisory"]);
        }
        if (node["pitch"]) {
            pitch.parse(node["pitch"]);
        }
        if (node["torque"]) {
            torque.parse(node["torque"]);
        }
    }
};

// Environment
struct Environment {
    double gravity{};               // Gravitational acceleration
    double air_density{};           // Density of air.
    double air_dyn_viscosity{};     // Dynamic viscosity of air.
    double air_pressure{};          // Atmospheric pressure of air
    double air_vapor_pressure{};    // Vapor pressure of fluid
    double weib_shape_parameter{};  // Shape factor of the Weibull wind distribution.
    double air_speed_sound{};       // Speed of sound in air.
    double shear_exp{};             // Shear exponent of the atmospheric boundary layer.
    double water_density{};         // Density of water.
    double water_dyn_viscosity{};   // Dynamic viscosity of water.
    double water_depth{};           // Water depth for offshore environment.
    double soil_shear_modulus{};    // Shear modulus of the soil.
    double soil_poisson{};          // Poisson ratio of the soil.
    double v_mean{};  // Average inflow wind speed. If different than 0, this will overwrite the V
                      // mean of the IEC wind class

    void parse(const YAML::Node& node) {
        gravity = node["gravity"].as<double>(0.);
        air_density = node["air_density"].as<double>(0.);
        air_dyn_viscosity = node["air_dyn_viscosity"].as<double>(0.);
        air_pressure = node["air_pressure"].as<double>(0.);
        air_vapor_pressure = node["air_vapor_pressure"].as<double>(0.);
        weib_shape_parameter = node["weib_shape_parameter"].as<double>(0.);
        air_speed_sound = node["air_speed_sound"].as<double>(0.);
        shear_exp = node["shear_exp"].as<double>(0.);
        water_density = node["water_density"].as<double>(0.);
        water_dyn_viscosity = node["water_dyn_viscosity"].as<double>(0.);
        water_depth = node["water_depth"].as<double>(0.);
        soil_shear_modulus = node["soil_shear_modulus"].as<double>(0.);
        soil_poisson = node["soil_poisson"].as<double>(0.);
        v_mean = node["v_mean"].as<double>(0.);
    }
};

// Bos
struct Bos {
    double plant_turbine_spacing{};  // Distance between turbines in the primary grid streamwise
                                     // direction in rotor diameters
    double plant_row_spacing{};      // Distance between turbine rows in the cross-wind direction in
                                 // rotor diameters
    double commissioning_pct{};       // Fraction of total BOS cost that is due to commissioning
    double decommissioning_pct{};     // Fraction of total BOS cost that is due to decommissioning
    double distance_to_substation{};  // Distance from centroid of plant to substation in km
    double distance_to_interconnection{};  // Distance from substation to grid connection in km
    double distance_to_landfall{};       // Distance from plant centroid to export cable landfall for
                                         // offshore plants
    double distance_to_site{};           // Distance from port to plant centroid for offshore plants
    double interconnect_voltage{};       // Voltage of cabling to grid interconnection
    double port_cost_per_month{};        // Monthly port rental fees
    double site_auction_price{};         // Cost to secure site lease
    double site_assessment_plan_cost{};  // Cost to do engineering plan for site assessment
    double site_assessment_cost{};       // Cost to execute site assessment
    double construction_operations_plan_cost{};  // Cost to do construction planning
    double boem_review_cost{};  // Cost for additional review by U.S. Dept of Interior Bureau of
                                // Ocean Energy Management (BOEM)
    double design_install_plan_cost{};  // Cost to do installation planning

    void parse(const YAML::Node& node) {
        plant_turbine_spacing = node["plant_turbine_spacing"].as<double>(0.);
        plant_row_spacing = node["plant_row_spacing"].as<double>(0.);
        commissioning_pct = node["commissioning_pct"].as<double>(0.);
        decommissioning_pct = node["decommissioning_pct"].as<double>(0.);
        distance_to_substation = node["distance_to_substation"].as<double>(0.);
        distance_to_interconnection = node["distance_to_interconnection"].as<double>(0.);
        distance_to_landfall = node["distance_to_landfall"].as<double>(0.);
        distance_to_site = node["distance_to_site"].as<double>(0.);
        interconnect_voltage = node["interconnect_voltage"].as<double>(0.);
        port_cost_per_month = node["port_cost_per_month"].as<double>(0.);
        site_auction_price = node["site_auction_price"].as<double>(0.);
        site_assessment_plan_cost = node["site_assessment_plan_cost"].as<double>(0.);
        site_assessment_cost = node["site_assessment_cost"].as<double>(0.);
        construction_operations_plan_cost = node["construction_operations_plan_cost"].as<double>(0.);
        boem_review_cost = node["boem_review_cost"].as<double>(0.);
        design_install_plan_cost = node["design_install_plan_cost"].as<double>(0.);
    }
};

// Costs
struct Costs {
    double wake_loss_factor{};  // Factor to model losses in annual energy production in a wind farm
                                // compared to the annual energy production at the turbine level
                                // (wakes mostly).
    double fixed_charge_rate{};  // Fixed charge rate to compute the levelized cost of energy. See
                                 // this for inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    double bos_per_kw{};   // Balance of stations costs expressed in USD per kW. See this for
                           // inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    double opex_per_kw{};  // Operational expenditures expressed in USD per kW. See this for
                           // inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    int turbine_number{};  // Number of turbines in the park, used to compute levelized cost of
                           // energy. Often wind parks are assumed of 600 MW. See this for
                           // inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    double labor_rate{};  // Hourly loaded wage per worker including all benefits and overhead.  This
                          // is currently only applied to steel, column structures.
    double painting_rate{};  // Cost per unit area for finishing and surface treatments.  This is
                             // currently only applied to steel, column structures.
    double blade_mass_cost_coeff{};         // Regression-based blade cost/mass ratio
    double hub_mass_cost_coeff{};           // Regression-based hub cost/mass ratio
    double pitch_system_mass_cost_coeff{};  // Regression-based pitch system cost/mass ratio
    double spinner_mass_cost_coeff{};       // Regression-based spinner cost/mass ratio
    double lss_mass_cost_coeff{};           // Regression-based low speed shaft cost/mass ratio
    double bearing_mass_cost_coeff{};       // Regression-based bearing cost/mass ratio
    double gearbox_mass_cost_coeff{};       // Regression-based gearbox cost/mass ratio
    double hss_mass_cost_coeff{};           // Regression-based high speed side cost/mass ratio
    double generator_mass_cost_coeff{};     // Regression-based generator cost/mass ratio
    double bedplate_mass_cost_coeff{};      // Regression-based bedplate cost/mass ratio
    double yaw_mass_cost_coeff{};           // Regression-based yaw system cost/mass ratio
    double converter_mass_cost_coeff{};     // Regression-based converter cost/mass ratio
    double transformer_mass_cost_coeff{};   // Regression-based transformer cost/mass ratio
    double hvac_mass_cost_coeff{};          // Regression-based HVAC system cost/mass ratio
    double cover_mass_cost_coeff{};         // Regression-based nacelle cover cost/mass ratio
    double elec_connec_machine_rating_cost_coeff{};  // Regression-based electrical plant connection
                                                     // cost/rating ratio
    double platforms_mass_cost_coeff{};  // Regression-based nacelle platform cost/mass ratio
    double tower_mass_cost_coeff{};      // Regression-based tower cost/mass ratio
    double controls_machine_rating_cost_coeff{};  // Regression-based controller and sensor system
                                                  // cost/rating ratio
    double crane_cost{};                          // crane cost if present
    double electricity_price{};  // Electricity price used to compute value in beyond lcoe metrics
    double
        reserve_margin_price{};  // Reserve margin price used to compute value in beyond lcoe metrics
    double capacity_credit{};    // Capacity credit used to compute value in beyond lcoe metrics
    double
        benchmark_price{};  // Benchmark price used to nondimensionalize value in beyond lcoe metrics

    void parse(const YAML::Node& node) {
        wake_loss_factor = node["wake_loss_factor"].as<double>(0.);
        fixed_charge_rate = node["fixed_charge_rate"].as<double>(0.);
        bos_per_kw = node["bos_per_kw"].as<double>(0.);
        opex_per_kw = node["opex_per_kw"].as<double>(0.);
        turbine_number = node["turbine_number"].as<int>(0);
        labor_rate = node["labor_rate"].as<double>(0.);
        painting_rate = node["painting_rate"].as<double>(0.);
        blade_mass_cost_coeff = node["blade_mass_cost_coeff"].as<double>(0.);
        hub_mass_cost_coeff = node["hub_mass_cost_coeff"].as<double>(0.);
        pitch_system_mass_cost_coeff = node["pitch_system_mass_cost_coeff"].as<double>(0.);
        spinner_mass_cost_coeff = node["spinner_mass_cost_coeff"].as<double>(0.);
        lss_mass_cost_coeff = node["lss_mass_cost_coeff"].as<double>(0.);
        bearing_mass_cost_coeff = node["bearing_mass_cost_coeff"].as<double>(0.);
        gearbox_mass_cost_coeff = node["gearbox_mass_cost_coeff"].as<double>(0.);
        hss_mass_cost_coeff = node["hss_mass_cost_coeff"].as<double>(0.);
        generator_mass_cost_coeff = node["generator_mass_cost_coeff"].as<double>(0.);
        bedplate_mass_cost_coeff = node["bedplate_mass_cost_coeff"].as<double>(0.);
        yaw_mass_cost_coeff = node["yaw_mass_cost_coeff"].as<double>(0.);
        converter_mass_cost_coeff = node["converter_mass_cost_coeff"].as<double>(0.);
        transformer_mass_cost_coeff = node["transformer_mass_cost_coeff"].as<double>(0.);
        hvac_mass_cost_coeff = node["hvac_mass_cost_coeff"].as<double>(0.);
        cover_mass_cost_coeff = node["cover_mass_cost_coeff"].as<double>(0.);
        elec_connec_machine_rating_cost_coeff =
            node["elec_connec_machine_rating_cost_coeff"].as<double>(0.);
        platforms_mass_cost_coeff = node["platforms_mass_cost_coeff"].as<double>(0.);
        tower_mass_cost_coeff = node["tower_mass_cost_coeff"].as<double>(0.);
        controls_machine_rating_cost_coeff =
            node["controls_machine_rating_cost_coeff"].as<double>(0.);
        crane_cost = node["crane_cost"].as<double>(0.);
        electricity_price = node["electricity_price"].as<double>(0.);
        reserve_margin_price = node["reserve_margin_price"].as<double>(0.);
        capacity_credit = node["capacity_credit"].as<double>(0.);
        benchmark_price = node["benchmark_price"].as<double>(0.);
    }
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
        comments = node["comments"].as<std::string>("");
        name = node["name"].as<std::string>("");
        if (node["assembly"]) {
            assembly.parse(node["assembly"]);
        }
        if (node["components"]) {
            components.parse(node["components"]);
        }
        if (node["airfoils"]) {
            for (const auto& item : node["airfoils"]) {
                Airfoils x;
                x.parse(item);
                airfoils.push_back(x);
            }
        }
        if (node["materials"]) {
            for (const auto& item : node["materials"]) {
                Materials x;
                x.parse(item);
                materials.push_back(x);
            }
        }
        if (node["control"]) {
            control.parse(node["control"]);
        }
        if (node["environment"]) {
            environment.parse(node["environment"]);
        }
        if (node["bos"]) {
            bos.parse(node["bos"]);
        }
        if (node["costs"]) {
            costs.parse(node["costs"]);
        }
    }
};
