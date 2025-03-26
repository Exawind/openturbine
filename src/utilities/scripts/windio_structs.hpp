
#pragma once

#include <string>
#include <variant>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace openturbine::wind_io {

// The field assembly includes nine entries that aim at describing the overall configuration of the wind turbine
struct Assembly {
    std::string turbine_class; // IEC wind class. The entry should be :code:`I`, :code:`II`, :code:`III`, or :code:`IV`.
    std::string turbulence_class; // IEC turbulence class of the wind turbine. The options are :code:`A`, :code:`B`, :code:`C`, and :code:`D`.
    std::string drivetrain; // Drivetrain configuration. This is intended to inform an automated interpreter of the yaml about the data specified in the field drivetrain
    std::string rotor_orientation; // Switch between :code:`upwind` and :code:`downwind` rotor configurations.
    int number_of_blades; // Integer setting the number of blades of the rotor
    double rotor_diameter; // Rotor diameter, defined as the sum of hub diameter and two times the blade length along its z axis, see blade
    double hub_height; // Height of the hub center over the ground (land-based) or the mean sea level (offshore)
    double rated_power; // Nameplate power of the turbine, i.e. the rated electrical output of the generator.
    double lifetime; // Turbine design lifetime in years.

    void parse(const YAML::Node& node) {
        turbine_class = node["turbine_class"] ? node["turbine_class"].as<std::string>() : "";
        turbulence_class = node["turbulence_class"] ? node["turbulence_class"].as<std::string>() : "";
        drivetrain = node["drivetrain"] ? node["drivetrain"].as<std::string>() : "";
        rotor_orientation = node["rotor_orientation"] ? node["rotor_orientation"].as<std::string>() : "";
        number_of_blades = node["number_of_blades"] ? node["number_of_blades"].as<int>() : 0;
        rotor_diameter = node["rotor_diameter"] ? node["rotor_diameter"].as<double>() : 0.;
        hub_height = node["hub_height"] ? node["hub_height"].as<double>() : 0.;
        rated_power = node["rated_power"] ? node["rated_power"].as<double>() : 0.;
        lifetime = node["lifetime"] ? node["lifetime"].as<double>() : 0.;
    }
};

// The array :code:`labels` specifies the names of the airfoils to be placed along the blade. The positions are specified in the field :code:`grid`. The two arrays must share the same length and to keep an airfoil constant along blade span, this must be defined twice. The :code:`labels` must match the :code:`names` of the airfoils listed in the top level :code:`airfoils`. In between airfoils, the recommended interpolation scheme for both coordinates and polars is the Piecewise Cubic Hermite Interpolating Polynomial (PCHIP), which is implemented in `Matlab <https://www.mathworks.com/help/matlab/ref/pchip.html>`_ and in the Python library `SciPy  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_. Optionally, the field :code:`rthick.values` can also be defined, see below.
struct AirfoilPosition {
    std::vector<double> grid;
    std::vector<std::string> labels;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        labels = node["labels"] ? node["labels"].as<std::vector<std::string>>() : std::vector<std::string>();
    }
};

// The array :code:`values` specifies the chord along blade span.
struct Chord {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// The array :code:`values` specifies the aerodynamic twist along blade span. Twist is generally positive toward blade root, and may become negative toward blade tip.
struct Twist {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// This parameter controls the airfoil position relative to the reference axis, by specifying the distance in meters along the chordline from the reference axis to the leading edge. 0 means that the airfoil is pinned at the leading edge, a positive offset means that the leading edge is upstream of the reference axis in local chordline coordinates, and a negative offset that the leading edge aft of the reference axis.
struct SectionOffsetX {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// This parameter controls the airfoil position relative to the reference axis, by specifying the chordline normal distance in meters from the reference axis. 0 means that the reference axis lies on the airfoil chordline, a positive offset means that the chordline is shifted in the direction of the suction side relative to the reference axis, and a negative offset that the section is shifted in the direction of the pressure side of the airfoil.
struct SectionOffsetY {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// This array has recently been added to windIO to overcome the uncertainty in the interpolated distribution of relative thickness along blade span. It should match the field :code:`airfoil_position`
struct Rthick {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// X
struct X {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Y
struct Y {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Z
struct Z {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// For the blade, the reference system is located at blade root, with z aligned with the pitch axis, x pointing towards the suction sides of the airfoils (standard prebend will be negative) and y pointing to the trailing edge (standard sweep will be positive). The blade coordinate system is the one of `BeamDyn  <https://openfast.readthedocs.io/en/master/source/user/beamdyn/input_files.html#beamdyn-primary-input-file>`_ of OpenFAST and it is shown in the figure below. The consequences of this reference system is that standard wind turbine blades have positive twist inboard and close to zero or even slightly negative twist outboard, zero or negative x values for standard prebent blades, and positive y values for backward swept blades. The blade main direction is expressed along z, and total blade length must be computed integrating the fields x, y, and z three-dimensionally. For tower and moniopile, :code:`x` is parallel to the ground pointing downwind, :code:`y` is parallel to the ground and to the rotor plane, and :code:`z` is perpendicular to the ground pointing upwards. Standard monopiles are only defined along :code:`z`.
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

// :code:`outer_shape_bem` consists of a dictionary containing the data for blade BEM-based aerodynamics.
struct OuterShapeBem {
    AirfoilPosition airfoil_position; // The array :code:`labels` specifies the names of the airfoils to be placed along the blade. The positions are specified in the field :code:`grid`. The two arrays must share the same length and to keep an airfoil constant along blade span, this must be defined twice. The :code:`labels` must match the :code:`names` of the airfoils listed in the top level :code:`airfoils`. In between airfoils, the recommended interpolation scheme for both coordinates and polars is the Piecewise Cubic Hermite Interpolating Polynomial (PCHIP), which is implemented in `Matlab <https://www.mathworks.com/help/matlab/ref/pchip.html>`_ and in the Python library `SciPy  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_. Optionally, the field :code:`rthick.values` can also be defined, see below.
    Chord chord; // The array :code:`values` specifies the chord along blade span.
    Twist twist; // The array :code:`values` specifies the aerodynamic twist along blade span. Twist is generally positive toward blade root, and may become negative toward blade tip.
    SectionOffsetX section_offset_x; // This parameter controls the airfoil position relative to the reference axis, by specifying the distance in meters along the chordline from the reference axis to the leading edge. 0 means that the airfoil is pinned at the leading edge, a positive offset means that the leading edge is upstream of the reference axis in local chordline coordinates, and a negative offset that the leading edge aft of the reference axis.
    SectionOffsetY section_offset_y; // This parameter controls the airfoil position relative to the reference axis, by specifying the chordline normal distance in meters from the reference axis. 0 means that the reference axis lies on the airfoil chordline, a positive offset means that the chordline is shifted in the direction of the suction side relative to the reference axis, and a negative offset that the section is shifted in the direction of the pressure side of the airfoil.
    Rthick rthick; // This array has recently been added to windIO to overcome the uncertainty in the interpolated distribution of relative thickness along blade span. It should match the field :code:`airfoil_position`
    ReferenceAxis reference_axis;

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
        if (node["section_offset_x"]) {
            section_offset_x.parse(node["section_offset_x"]);
        }
        if (node["section_offset_y"]) {
            section_offset_y.parse(node["section_offset_y"]);
        }
        if (node["rthick"]) {
            rthick.parse(node["rthick"]);
        }
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
    }
};

// StiffnessMatrix
struct StiffnessMatrix {
    std::vector<double> grid;
    std::vector<double> k11; // Distribution of the K11 element of the stiffness matrix along blade span. K11 corresponds to the shear stiffness along the x axis (in a blade, x points to the trailing edge)
    std::vector<double> k22; // Distribution of the K22 element of the stiffness matrix along blade span. K22 corresponds to the shear stiffness along the y axis (in a blade, y points to the suction side)
    std::vector<double> k33; // Distribution of the K33 element of the stiffness matrix along blade span. K33 corresponds to the axial stiffness along the z axis (in a blade, z runs along the span and points to the tip)
    std::vector<double> k44; // Distribution of the K44 element of the stiffness matrix along blade span. K44 corresponds to the bending stiffness around the x axis (in a blade, x points to the trailing edge and K44 corresponds to the flapwise stiffness)
    std::vector<double> k55; // Distribution of the K55 element of the stiffness matrix along blade span. K55 corresponds to the bending stiffness around the y axis (in a blade, y points to the suction side and K55 corresponds to the edgewise stiffness)
    std::vector<double> k66; // Distribution of K66 element of the stiffness matrix along blade span. K66 corresponds to the torsional stiffness along the z axis (in a blade, z runs along the span and points to the tip)
    std::vector<double> k12; // Distribution of the K12 element of the stiffness matrix along blade span. K12 is a cross term between shear terms
    std::vector<double> k13; // Distribution of the K13 element of the stiffness matrix along blade span. K13 is a cross term shear - axial
    std::vector<double> k14; // Distribution of the K14 element of the stiffness matrix along blade span. K14 is a cross term shear - bending
    std::vector<double> k15; // Distribution of the K15 element of the stiffness matrix along blade span. K15 is a cross term shear - bending
    std::vector<double> k16; // Distribution of the K16 element of the stiffness matrix along blade span. K16 is a cross term shear - torsion
    std::vector<double> k23; // Distribution of the K23 element of the stiffness matrix along blade span. K23 is a cross term shear - axial
    std::vector<double> k24; // Distribution of the K24 element of the stiffness matrix along blade span. K24 is a cross term shear - bending
    std::vector<double> k25; // Distribution of the K25 element of the stiffness matrix along blade span. K25 is a cross term shear - bending
    std::vector<double> k26; // Distribution of the K26 element of the stiffness matrix along blade span. K26 is a cross term shear - torsion
    std::vector<double> k34; // Distribution of the K34 element of the stiffness matrix along blade span. K34 is a cross term axial - bending
    std::vector<double> k35; // Distribution of the K35 element of the stiffness matrix along blade span. K35 is a cross term axial - bending
    std::vector<double> k36; // Distribution of the K36 element of the stiffness matrix along blade span. K36 is a cross term axial - torsion
    std::vector<double> k45; // Distribution of the K45 element of the stiffness matrix along blade span. K45 is a cross term flapwise bending - edgewise bending
    std::vector<double> k46; // Distribution of the K46 element of the stiffness matrix along blade span. K46 is a cross term flapwise bending - torsion
    std::vector<double> k56; // Distribution of the K56 element of the stiffness matrix along blade span. K56 is a cross term edgewise bending - torsion

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        k11 = node["k11"] ? node["k11"].as<std::vector<double>>() : std::vector<double>();
        k22 = node["k22"] ? node["k22"].as<std::vector<double>>() : std::vector<double>();
        k33 = node["k33"] ? node["k33"].as<std::vector<double>>() : std::vector<double>();
        k44 = node["k44"] ? node["k44"].as<std::vector<double>>() : std::vector<double>();
        k55 = node["k55"] ? node["k55"].as<std::vector<double>>() : std::vector<double>();
        k66 = node["k66"] ? node["k66"].as<std::vector<double>>() : std::vector<double>();
        k12 = node["k12"] ? node["k12"].as<std::vector<double>>() : std::vector<double>();
        k13 = node["k13"] ? node["k13"].as<std::vector<double>>() : std::vector<double>();
        k14 = node["k14"] ? node["k14"].as<std::vector<double>>() : std::vector<double>();
        k15 = node["k15"] ? node["k15"].as<std::vector<double>>() : std::vector<double>();
        k16 = node["k16"] ? node["k16"].as<std::vector<double>>() : std::vector<double>();
        k23 = node["k23"] ? node["k23"].as<std::vector<double>>() : std::vector<double>();
        k24 = node["k24"] ? node["k24"].as<std::vector<double>>() : std::vector<double>();
        k25 = node["k25"] ? node["k25"].as<std::vector<double>>() : std::vector<double>();
        k26 = node["k26"] ? node["k26"].as<std::vector<double>>() : std::vector<double>();
        k34 = node["k34"] ? node["k34"].as<std::vector<double>>() : std::vector<double>();
        k35 = node["k35"] ? node["k35"].as<std::vector<double>>() : std::vector<double>();
        k36 = node["k36"] ? node["k36"].as<std::vector<double>>() : std::vector<double>();
        k45 = node["k45"] ? node["k45"].as<std::vector<double>>() : std::vector<double>();
        k46 = node["k46"] ? node["k46"].as<std::vector<double>>() : std::vector<double>();
        k56 = node["k56"] ? node["k56"].as<std::vector<double>>() : std::vector<double>();
    }
};

// InertiaMatrix
struct InertiaMatrix {
    std::vector<double> grid;
    std::vector<double> mass; // Mass per unit length along the beam, expressed in kilogram per meter
    std::vector<double> cm_x; // Distance between the reference axis and the center of mass along the x axis
    std::vector<double> cm_y; // Distance between the reference axis and the center of mass along the y axis
    std::vector<double> i_edge; // Edgewise mass moment of inertia per unit span (around y axis)
    std::vector<double> i_flap; // Flapwise mass moment of inertia per unit span (around x axis)
    std::vector<double> i_plr; // Polar moment of inertia per unit span (around z axis). Please note that for beam-like structures iplr must be equal to iedge plus iflap.
    std::vector<double> i_cp; // Sectional cross-product of inertia per unit span (cross term x y)

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        mass = node["mass"] ? node["mass"].as<std::vector<double>>() : std::vector<double>();
        cm_x = node["cm_x"] ? node["cm_x"].as<std::vector<double>>() : std::vector<double>();
        cm_y = node["cm_y"] ? node["cm_y"].as<std::vector<double>>() : std::vector<double>();
        i_edge = node["i_edge"] ? node["i_edge"].as<std::vector<double>>() : std::vector<double>();
        i_flap = node["i_flap"] ? node["i_flap"].as<std::vector<double>>() : std::vector<double>();
        i_plr = node["i_plr"] ? node["i_plr"].as<std::vector<double>>() : std::vector<double>();
        i_cp = node["i_cp"] ? node["i_cp"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Structural damping of the beam. For now, viscous damping is included
struct StructuralDamping {
    std::vector<double> mu; // Six damping coefficients to model viscous damping, where the damping forces are proportional to the strain rate

    void parse(const YAML::Node& node) {
        mu = node["mu"] ? node["mu"].as<std::vector<double>>() : std::vector<double>();
    }
};

// PointMass
struct PointMass {
    std::vector<double> grid;
    std::vector<double> mass; // Point masses distributed along the beam, expressed in kilogram. These can be used to model features such as flanges.

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        mass = node["mass"] ? node["mass"].as<std::vector<double>>() : std::vector<double>();
    }
};

// The equivalent elastic properties of a beam are defined in :code:`elastic_properties_mb`. Here, 6x6 stiffness and mass matrices are defined. Out of 36 entries of the matrices, given the symmetry, the yaml file requires the definition of only 21 values as inputs for the stiffness matrix, whereas the inertia matrix is defined in terms of unit mass, coordinates of the center of mass, and mass moments of inertia.
struct ElasticPropertiesMb {
    ReferenceAxis reference_axis;
    Twist twist; // The array :code:`values` specifies the structural twist along blade span.
    StiffnessMatrix stiffness_matrix;
    InertiaMatrix inertia_matrix;
    StructuralDamping structural_damping; // Structural damping of the beam. For now, viscous damping is included
    PointMass point_mass;

    void parse(const YAML::Node& node) {
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["twist"]) {
            twist.parse(node["twist"]);
        }
        if (node["stiffness_matrix"]) {
            stiffness_matrix.parse(node["stiffness_matrix"]);
        }
        if (node["inertia_matrix"]) {
            inertia_matrix.parse(node["inertia_matrix"]);
        }
        if (node["structural_damping"]) {
            structural_damping.parse(node["structural_damping"]);
        }
        if (node["point_mass"]) {
            point_mass.parse(node["point_mass"]);
        }
    }
};

// Root
struct Root {
    double d_f; // Diameter of the fastener, default is M30, so 0.03 meters
    double sigma_max; // Max stress on bolt

    void parse(const YAML::Node& node) {
        d_f = node["d_f"] ? node["d_f"].as<double>() : 0.;
        sigma_max = node["sigma_max"] ? node["sigma_max"].as<double>() : 0.;
    }
};

// Non-dimensional location of the point along the non-dimensional arc length
struct StartNdArc {
    std::vector<double> grid;
    std::vector<double> values; // Grid along an arc length, expressed non dimensionally where 0 is the trailing edge on the suction side and 1 is the trailing edge on the pressure side. The arc starts and ends at the mid-point of the trailing edge in the case of open trailing edges.

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Non-dimensional location of the point along the non-dimensional arc length
struct EndNdArc {
    std::vector<double> grid;
    std::vector<double> values; // Grid along an arc length, expressed non dimensionally where 0 is the trailing edge on the suction side and 1 is the trailing edge on the pressure side. The arc starts and ends at the mid-point of the trailing edge in the case of open trailing edges.

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// The rotation defines the angle between the chord line and the y axis and it has the opposite sign of the twist. For shear webs perpendicular to the chord line in the section(s) where twist is zero, it is easiest to simply use the keyword fixed :code:`twist`. Blades with straight shear webs will always have the field rotation equal to the twist plus/minus a constant angle and, assuming a non-swept blade (zero values in the blade y reference axis), a linear field offset.
struct Rotation {
    std::vector<double> grid;
    std::vector<double> values;
    std::string fixed; // Name of the layer to lock the edge

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
        fixed = node["fixed"] ? node["fixed"].as<std::string>() : "";
    }
};

// Dimensional offset in respect to the reference axis along the x axis, which is the chord line rotated by a user-defined angle. Negative values move the midpoint towards the leading edge, positive towards the trailing edge
struct OffsetYRefAxis {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Webs
struct Webs {
    std::string name; // String that identifies the web.
    StartNdArc start_nd_arc;
    EndNdArc end_nd_arc;
    Rotation rotation;
    OffsetYRefAxis offset_y_ref_axis;

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        if (node["start_nd_arc"]) {
            start_nd_arc.parse(node["start_nd_arc"]);
        }
        if (node["end_nd_arc"]) {
            end_nd_arc.parse(node["end_nd_arc"]);
        }
        if (node["rotation"]) {
            rotation.parse(node["rotation"]);
        }
        if (node["offset_y_ref_axis"]) {
            offset_y_ref_axis.parse(node["offset_y_ref_axis"]);
        }
    }
};

// Dimensional thickness of the laminate, expressed in meters. This value is modeled constant along the section. To define ply drops along the 2D surface, the user is therefore required to define multiple layers, possibly ply by ply when many ply drops exist.
struct Thickness {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// In addition or in alternative to the dimensional thickness, the discrete number of plies of a composite laminate can be defined by the user. Notably, the ply thickness is a material property (not a layer property) and it is defined in the top-level field :code:`materials`.
struct NPlies {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// For composite laminates, the orientation of the fibers in degrees can be specified. Looking from blade root, positive angles represent a rotation of the fibers towards the leading edge of the blade. Note that the angles are with respect to the cross section local reference system, not the reference system at blade root.
struct FiberOrientation {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// The field width defines the width in meters along the arc of the layer.
struct Width {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Non-dimensional location of the point along the non-dimensional arc length
struct MidpointNdArc {
    std::vector<double> grid;
    std::vector<double> values; // Grid along an arc length, expressed non dimensionally where 0 is the trailing edge on the suction side and 1 is the trailing edge on the pressure side. The arc starts and ends at the mid-point of the trailing edge in the case of open trailing edges.

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Layers
struct Layers {
    std::string name; // String that identifies the layer.
    std::string material; // String that identifies the material of the layer. The material and its properties must be defined in the top-level :code:`materials`.
    std::string web; // web to which the layer is associated to, only to be defined for web layers
    Thickness thickness; // Dimensional thickness of the laminate, expressed in meters. This value is modeled constant along the section. To define ply drops along the 2D surface, the user is therefore required to define multiple layers, possibly ply by ply when many ply drops exist.
    NPlies n_plies; // In addition or in alternative to the dimensional thickness, the discrete number of plies of a composite laminate can be defined by the user. Notably, the ply thickness is a material property (not a layer property) and it is defined in the top-level field :code:`materials`.
    FiberOrientation fiber_orientation; // For composite laminates, the orientation of the fibers in degrees can be specified. Looking from blade root, positive angles represent a rotation of the fibers towards the leading edge of the blade. Note that the angles are with respect to the cross section local reference system, not the reference system at blade root.
    Width width; // The field width defines the width in meters along the arc of the layer.
    MidpointNdArc midpoint_nd_arc;
    std::string side; // The field side is string that can be either :code:`suction` or :code:`pressure`, defining the side where a layer is defined.
    StartNdArc start_nd_arc;
    EndNdArc end_nd_arc;
    Rotation rotation;
    OffsetYRefAxis offset_y_ref_axis;

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        material = node["material"] ? node["material"].as<std::string>() : "";
        web = node["web"] ? node["web"].as<std::string>() : "";
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
        side = node["side"] ? node["side"].as<std::string>() : "";
        if (node["start_nd_arc"]) {
            start_nd_arc.parse(node["start_nd_arc"]);
        }
        if (node["end_nd_arc"]) {
            end_nd_arc.parse(node["end_nd_arc"]);
        }
        if (node["rotation"]) {
            rotation.parse(node["rotation"]);
        }
        if (node["offset_y_ref_axis"]) {
            offset_y_ref_axis.parse(node["offset_y_ref_axis"]);
        }
    }
};

// This is a spanwise joint along the blade, usually adopted to ease transportation constraints
struct Joint {
    double position; // Spanwise position of the segmentation joint.
    double mass; // Mass of the joint.
    double cost; // Cost of the joint.
    std::string bolt; // Bolt size for the blade bolted joint
    double nonmaterial_cost; // Cost of the joint not from materials.
    std::string reinforcement_layer_ss; // Layer identifier for the joint reinforcement on the suction side
    std::string reinforcement_layer_ps; // Layer identifier for the joint reinforcement on the pressure side

    void parse(const YAML::Node& node) {
        position = node["position"] ? node["position"].as<double>() : 0.;
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        cost = node["cost"] ? node["cost"].as<double>() : 0.;
        bolt = node["bolt"] ? node["bolt"].as<std::string>() : "";
        nonmaterial_cost = node["nonmaterial_cost"] ? node["nonmaterial_cost"].as<double>() : 0.;
        reinforcement_layer_ss = node["reinforcement_layer_ss"] ? node["reinforcement_layer_ss"].as<std::string>() : "";
        reinforcement_layer_ps = node["reinforcement_layer_ps"] ? node["reinforcement_layer_ps"].as<std::string>() : "";
    }
};

// The field :code:`internal_structure_2d_fem` contains the data to describe the internal structure of standard wind turbine blades. This is a fairly sophisticated process and the ontology proposed in this work supports different definitions. On the top level, the field :code:`internal_structure_2d_fem` has three sub-components, namely the :code:`reference_axis`, which is usually defined equal to the :code:`reference_axis` in the field :code:`outer_shape_bem`, the :code:`webs`, where the positions of the shear webs are defined, and the :code:`layers`, which describe all internal layers in terms of :code:`name`, :code:`material`, :code:`thickness`, number of plies :code:`n_plies`, :code:`fiber_orientation` (for composites), and position in the two-dimensional sections. Recently, the fields :code:`joint` and :code:`root` were added to support blades that are segmented spanwise and details about the blade root bolting.
struct InternalStructure2DFem {
    Root root;
    ReferenceAxis reference_axis;
    std::vector<Webs> webs; // The field :code:`webs` consists of a list of entries, each representing a shear web defined in terms of :code:`name` and position.
    std::vector<Layers> layers; // The sub-field :code:`layers` defines the material layers of the wind turbine blade. In most cases, these are layers of composite materials. The most convenient approach to define the position of spar caps mimics the definition of the shear webs, adding the width and side that define the width of the layer in meters and the side where the layer is located, either “pressure” or “suction”. Layers that wrap the entire cross section, such as paint or outer shell skin, can be defined with :code:`start_nd_arc` set to 0 and :code:`end_nd_arc` set to 1. To define reinforcements, the best way is usually to define the width, in meters, and the midpoint, named :code:`midpoint_nd_arc` and defined nondimensional between 0 and 1. Converters should be able to look for the leading edge, marked as LE. Similar combinations can be constructed with the combination of :code:`width` and :code:`start_nd_arc` or :code:`end_nd_arc`. Finally, for composite layers belonging to the shear webs, a tag :code:`web` should contain the name of the web. The layers are then modeled from leading edge to trailing edge in the order they were specified.
    Joint joint; // This is a spanwise joint along the blade, usually adopted to ease transportation constraints

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

// The component :code:`blade` includes three subcomponents, namely :code:`outer_shape_bem`, :code:`elastic_properties_mb`, and :code:`internal_structure_2d_fem`. A fourth and a fifth subfields :code:`cfd_geometry` and :code:`3D_fem` will be added to support higher fidelity modeling of the rotor. All distributed quantities, such as blade chord or the thickness of a structural component, are expressed in terms of pair arrays :code:`grid` and :code:`values`, which must have a minimum length of two elements and the same size. :code:`grid` is defined nondimensional between 0 (root) and 1 (tip) along the, possibly curved, :code:`reference_axis`.
struct Blade {
    OuterShapeBem outer_shape_bem; // :code:`outer_shape_bem` consists of a dictionary containing the data for blade BEM-based aerodynamics.
    ElasticPropertiesMb elastic_properties_mb;
    InternalStructure2DFem internal_structure_2d_fem; // The field :code:`internal_structure_2d_fem` contains the data to describe the internal structure of standard wind turbine blades. This is a fairly sophisticated process and the ontology proposed in this work supports different definitions. On the top level, the field :code:`internal_structure_2d_fem` has three sub-components, namely the :code:`reference_axis`, which is usually defined equal to the :code:`reference_axis` in the field :code:`outer_shape_bem`, the :code:`webs`, where the positions of the shear webs are defined, and the :code:`layers`, which describe all internal layers in terms of :code:`name`, :code:`material`, :code:`thickness`, number of plies :code:`n_plies`, :code:`fiber_orientation` (for composites), and position in the two-dimensional sections. Recently, the fields :code:`joint` and :code:`root` were added to support blades that are segmented spanwise and details about the blade root bolting.

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
    double diameter; // This is the outer diameter of the hub. It is also the diameter of the circle centered at the rotor apex and connecting the blade root centers.
    double cone_angle; // Rotor precone angle, defined positive for both upwind and downwind rotors.
    double cd; // Equivalent drag coefficient to compute the aerodynamic forces generated on the hub.

    void parse(const YAML::Node& node) {
        diameter = node["diameter"] ? node["diameter"].as<double>() : 0.;
        cone_angle = node["cone_angle"] ? node["cone_angle"].as<double>() : 0.;
        cd = node["cd"] ? node["cd"].as<double>() : 0.;
    }
};

// ElasticPropertiesMb_1
struct ElasticPropertiesMb_1 {
    double mass; // Mass of the component modeled as a point
    std::vector<double> inertia; // Mass moment of inertia of the component modeled as a point
    std::vector<double> location; // Location of the point mass with respect to the coordinate system
    std::string coordinate_system; // Coordinate system used to define the location of the point mass

    void parse(const YAML::Node& node) {
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        inertia = node["inertia"] ? node["inertia"].as<std::vector<double>>() : std::vector<double>();
        location = node["location"] ? node["location"].as<std::vector<double>>() : std::vector<double>();
        coordinate_system = node["coordinate_system"] ? node["coordinate_system"].as<std::string>() : "";
    }
};

// Hub
struct Hub {
    OuterShapeBem_1 outer_shape_bem;
    ElasticPropertiesMb_1 elastic_properties_mb; // Point mass modeling the full hub system, which includes the hub, the spinner, the blade bearings, the pitch actuators, the cabling, ....

    void parse(const YAML::Node& node) {
        if (node["outer_shape_bem"]) {
            outer_shape_bem.parse(node["outer_shape_bem"]);
        }
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
    }
};

// Geometrical metrics describing the drivetrain. Currently, these are inputs to WISDEM specific drivetrain sizing tool, DrivetrainSE
struct OuterShapeBem_2 {
    double uptilt; // Tilt angle of the nacelle, always defined positive.
    double distance_tt_hub; // Vertical distance between the tower top and the hub center.
    double distance_hub_mb; // Distance from hub flange to first main bearing along shaft.
    double distance_mb_mb; // Distance from first to second main bearing along shaft.
    double overhang; // Horizontal distance between the tower axis and the rotor apex.

    void parse(const YAML::Node& node) {
        uptilt = node["uptilt"] ? node["uptilt"].as<double>() : 0.;
        distance_tt_hub = node["distance_tt_hub"] ? node["distance_tt_hub"].as<double>() : 0.;
        distance_hub_mb = node["distance_hub_mb"] ? node["distance_hub_mb"].as<double>() : 0.;
        distance_mb_mb = node["distance_mb_mb"] ? node["distance_mb_mb"].as<double>() : 0.;
        overhang = node["overhang"] ? node["overhang"].as<double>() : 0.;
    }
};

// ElasticPropertiesMb_2
struct ElasticPropertiesMb_2 {
    double mass; // Mass of the component modeled as a point
    std::vector<double> inertia; // Mass moment of inertia of the component modeled as a point
    std::vector<double> location; // Location of the point mass with respect to the coordinate system
    std::string coordinate_system; // Coordinate system used to define the location of the point mass

    void parse(const YAML::Node& node) {
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        inertia = node["inertia"] ? node["inertia"].as<std::vector<double>>() : std::vector<double>();
        location = node["location"] ? node["location"].as<std::vector<double>>() : std::vector<double>();
        coordinate_system = node["coordinate_system"] ? node["coordinate_system"].as<std::string>() : "";
    }
};

// Inputs describing the gearbox, when present
struct Gearbox {
    double gear_ratio; // Gear ratio of the drivetrain. Set it to 1 for direct drive machines.
    double length; // User input override of gearbox length along shaft, only used when using gearbox_mass_user is > 0
    double radius; // User input override of gearbox radius, only used when using gearbox_mass_user is > 0
    double mass; // User input override of gearbox mass
    double efficiency; // Efficiency of the gearbox system.
    double damping_ratio; // Damping ratio for the drivetrain system
    std::string gear_configuration; // 3-letter string of Es or Ps to denote epicyclic or parallel gear configuration
    std::vector<int> planet_numbers; // Number of planets for epicyclic stages (use 0 for parallel)
    ElasticPropertiesMb_2 elastic_properties_mb; // Point mass modeling the gearbox

    void parse(const YAML::Node& node) {
        gear_ratio = node["gear_ratio"] ? node["gear_ratio"].as<double>() : 0.;
        length = node["length"] ? node["length"].as<double>() : 0.;
        radius = node["radius"] ? node["radius"].as<double>() : 0.;
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        efficiency = node["efficiency"] ? node["efficiency"].as<double>() : 0.;
        damping_ratio = node["damping_ratio"] ? node["damping_ratio"].as<double>() : 0.;
        gear_configuration = node["gear_configuration"] ? node["gear_configuration"].as<std::string>() : "";
        planet_numbers = node["planet_numbers"] ? node["planet_numbers"].as<std::vector<int>>() : std::vector<int>();
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
    }
};

// Inputs describing the low speed shaft
struct Lss {
    double length; // Length of the low speed shaft
    std::vector<double> diameter; // Diameter of the low speed shaft at beginning (generator/gearbox) and end (hub) points
    std::vector<double> wall_thickness; // Thickness of the low speed shaft at beginning (generator/gearbox) and end (hub) points
    std::string material; // Material name identifier
    ElasticPropertiesMb elastic_properties_mb; // Beam modelling the low speed shaft

    void parse(const YAML::Node& node) {
        length = node["length"] ? node["length"].as<double>() : 0.;
        diameter = node["diameter"] ? node["diameter"].as<std::vector<double>>() : std::vector<double>();
        wall_thickness = node["wall_thickness"] ? node["wall_thickness"].as<std::vector<double>>() : std::vector<double>();
        material = node["material"] ? node["material"].as<std::string>() : "";
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
    }
};

// Inputs describing the high speed shaft, when present
struct Hss {
    double length; // Length of the high speed shaft
    std::vector<double> diameter; // Diameter of the high speed shaft at beginning (generator) and end (generator) points
    std::vector<double> wall_thickness; // Thickness of the high speed shaft at beginning (generator) and end (generator) points
    std::string material; // Material name identifier
    ElasticPropertiesMb elastic_properties_mb; // Beam modelling the high speed shaft

    void parse(const YAML::Node& node) {
        length = node["length"] ? node["length"].as<double>() : 0.;
        diameter = node["diameter"] ? node["diameter"].as<std::vector<double>>() : std::vector<double>();
        wall_thickness = node["wall_thickness"] ? node["wall_thickness"].as<std::vector<double>>() : std::vector<double>();
        material = node["material"] ? node["material"].as<std::string>() : "";
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
    }
};

// Inputs describing the nose/turret at beginning (bedplate) and end (main bearing) points
struct Nose {
    std::vector<double> diameter; // Diameter of the nose/turret at beginning (bedplate) and end (main bearing) points
    std::vector<double> wall_thickness; // Thickness of the nose/turret at beginning (bedplate) and end (main bearing) points
    ElasticPropertiesMb elastic_properties_mb; // Beam modelling the nose

    void parse(const YAML::Node& node) {
        diameter = node["diameter"] ? node["diameter"].as<std::vector<double>>() : std::vector<double>();
        wall_thickness = node["wall_thickness"] ? node["wall_thickness"].as<std::vector<double>>() : std::vector<double>();
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
    }
};

// Thickness of the hollow elliptical bedplate used in direct drive configurations
struct WallThickness {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Inputs describing the hollow elliptical bedplate used in direct drive configurations
struct Bedplate {
    WallThickness wall_thickness; // Thickness of the hollow elliptical bedplate used in direct drive configurations
    double flange_width; // Bedplate I-beam flange width used in geared configurations
    double flange_thickness; // Bedplate I-beam flange thickness used in geared configurations
    double web_thickness; // Bedplate I-beam web thickness used in geared configurations
    std::string bedplate_material; // Material name identifier

    void parse(const YAML::Node& node) {
        if (node["wall_thickness"]) {
            wall_thickness.parse(node["wall_thickness"]);
        }
        flange_width = node["flange_width"] ? node["flange_width"].as<double>() : 0.;
        flange_thickness = node["flange_thickness"] ? node["flange_thickness"].as<double>() : 0.;
        web_thickness = node["web_thickness"] ? node["web_thickness"].as<double>() : 0.;
        bedplate_material = node["bedplate_material"] ? node["bedplate_material"].as<std::string>() : "";
    }
};

// ElasticPropertiesMb_3
struct ElasticPropertiesMb_3 {
    double mass; // Mass of the component modeled as a point
    std::vector<double> inertia; // Mass moment of inertia of the component modeled as a point
    std::vector<double> location; // Location of the point mass with respect to the coordinate system
    std::string coordinate_system; // Coordinate system used to define the location of the point mass

    void parse(const YAML::Node& node) {
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        inertia = node["inertia"] ? node["inertia"].as<std::vector<double>>() : std::vector<double>();
        location = node["location"] ? node["location"].as<std::vector<double>>() : std::vector<double>();
        coordinate_system = node["coordinate_system"] ? node["coordinate_system"].as<std::string>() : "";
    }
};

// Inputs describing all other drivetrain components, the assembly of brake, hvac, converter, transformer, and main bearings
struct OtherComponents {
    double brake_mass; // Override regular regression-based calculation of brake mass with this value
    double hvac_mass_coefficient; // Regression-based scaling coefficient on machine rating to get HVAC system mass
    double converter_mass; // Override regular regression-based calculation of converter mass with this value
    double transformer_mass; // Override regular regression-based calculation of transformer mass with this value
    std::string mb1type; // Type of bearing for first main bearing
    std::string mb2type; // Type of bearing for second main bearing
    bool uptower; // If power electronics are located uptower (True) or at tower base (False)
    ElasticPropertiesMb_3 elastic_properties_mb; // Point mass modeling the assembly of brake, hvac, converter, transformer, and main bearings

    void parse(const YAML::Node& node) {
        brake_mass = node["brake_mass"] ? node["brake_mass"].as<double>() : 0.;
        hvac_mass_coefficient = node["hvac_mass_coefficient"] ? node["hvac_mass_coefficient"].as<double>() : 0.;
        converter_mass = node["converter_mass"] ? node["converter_mass"].as<double>() : 0.;
        transformer_mass = node["transformer_mass"] ? node["transformer_mass"].as<double>() : 0.;
        mb1type = node["mb1type"] ? node["mb1type"].as<std::string>() : "";
        mb2type = node["mb2type"] ? node["mb2type"].as<std::string>() : "";
        uptower = node["uptower"] ? node["uptower"].as<bool>() : false;
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
    }
};

// User input override of generator rpm-efficiency values, with rpm as grid input and eff as values input
struct RpmEfficiency {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// ElasticPropertiesMb_4
struct ElasticPropertiesMb_4 {
    double mass; // Mass of the component modeled as a point
    std::vector<double> inertia; // Mass moment of inertia of the component modeled as a point
    std::vector<double> location; // Location of the point mass with respect to the coordinate system
    std::string coordinate_system; // Coordinate system used to define the location of the point mass

    void parse(const YAML::Node& node) {
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        inertia = node["inertia"] ? node["inertia"].as<std::vector<double>>() : std::vector<double>();
        location = node["location"] ? node["location"].as<std::vector<double>>() : std::vector<double>();
        coordinate_system = node["coordinate_system"] ? node["coordinate_system"].as<std::string>() : "";
    }
};

// Generator
struct Generator {
    double length; // Length of generator along the shaft
    double radius; // User input override of generator radius, only used when using simple generator scaling
    double mass; // User input override of generator mass, only used when using simple generator mass scaling
    RpmEfficiency rpm_efficiency; // User input override of generator rpm-efficiency values, with rpm as grid input and eff as values input
    double mass_coefficient; // When not doing a detailed generator design, use a simplified approach to generator scaling. This input allows for overriding of the regression-based scaling coefficient to obtain generator mass
    std::string type;
    double b_r; // Words
    double p_fe0e; // Words
    double p_fe0h; // Words
    double s_n; // Words
    double s_nmax; // Words
    double alpha_p; // Words
    double b_r_tau_r; // Words
    double b_ro; // Words
    double b_s_tau_s; // Words
    double b_so; // Words
    double cofi; // Words
    double freq; // Words
    double h_i; // Words
    double h_sy0; // Words
    double h_w; // Words
    double k_fes; // Words
    double k_fillr; // Words
    double k_fills; // Words
    double k_s; // Words
    int m; // Words
    double mu_0; // Permittivity of free space
    double mu_r; // Words
    double p; // Words
    double phi; // Words
    int q1; // Words
    int q3; // Words
    double ratio_mw2pp; // Words
    double resist_cu; // Resistivity of copper
    double sigma; // Maximum allowable shear stress
    double y_tau_p; // Words
    double y_tau_pr; // Words
    double i_0; // Words
    double d_r; // Words
    double h_m; // Words
    double h_0; // Words
    double h_s; // Words
    double len_s; // Words
    double n_r; // Words
    double rad_ag; // Words
    double t_wr; // Words
    double n_s; // Words
    double b_st; // Words
    double d_s; // Words
    double t_ws; // Words
    double rho_copper; // Copper density
    double rho_fe; // Structural steel density
    double rho_fes; // Electrical steel density
    double rho_pm; // Permanent magnet density
    double c_cu; // Copper cost
    double c_fe; // Structural steel cost
    double c_fes; // Electrical steel cost
    double c_pm; // Permanent magnet cost
    ElasticPropertiesMb_4 elastic_properties_mb; // Point mass modeling the generator

    void parse(const YAML::Node& node) {
        length = node["length"] ? node["length"].as<double>() : 0.;
        radius = node["radius"] ? node["radius"].as<double>() : 0.;
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        if (node["rpm_efficiency"]) {
            rpm_efficiency.parse(node["rpm_efficiency"]);
        }
        mass_coefficient = node["mass_coefficient"] ? node["mass_coefficient"].as<double>() : 0.;
        type = node["type"] ? node["type"].as<std::string>() : "";
        b_r = node["b_r"] ? node["b_r"].as<double>() : 0.;
        p_fe0e = node["p_fe0e"] ? node["p_fe0e"].as<double>() : 0.;
        p_fe0h = node["p_fe0h"] ? node["p_fe0h"].as<double>() : 0.;
        s_n = node["s_n"] ? node["s_n"].as<double>() : 0.;
        s_nmax = node["s_nmax"] ? node["s_nmax"].as<double>() : 0.;
        alpha_p = node["alpha_p"] ? node["alpha_p"].as<double>() : 0.;
        b_r_tau_r = node["b_r_tau_r"] ? node["b_r_tau_r"].as<double>() : 0.;
        b_ro = node["b_ro"] ? node["b_ro"].as<double>() : 0.;
        b_s_tau_s = node["b_s_tau_s"] ? node["b_s_tau_s"].as<double>() : 0.;
        b_so = node["b_so"] ? node["b_so"].as<double>() : 0.;
        cofi = node["cofi"] ? node["cofi"].as<double>() : 0.;
        freq = node["freq"] ? node["freq"].as<double>() : 0.;
        h_i = node["h_i"] ? node["h_i"].as<double>() : 0.;
        h_sy0 = node["h_sy0"] ? node["h_sy0"].as<double>() : 0.;
        h_w = node["h_w"] ? node["h_w"].as<double>() : 0.;
        k_fes = node["k_fes"] ? node["k_fes"].as<double>() : 0.;
        k_fillr = node["k_fillr"] ? node["k_fillr"].as<double>() : 0.;
        k_fills = node["k_fills"] ? node["k_fills"].as<double>() : 0.;
        k_s = node["k_s"] ? node["k_s"].as<double>() : 0.;
        m = node["m"] ? node["m"].as<int>() : 0;
        mu_0 = node["mu_0"] ? node["mu_0"].as<double>() : 0.;
        mu_r = node["mu_r"] ? node["mu_r"].as<double>() : 0.;
        p = node["p"] ? node["p"].as<double>() : 0.;
        phi = node["phi"] ? node["phi"].as<double>() : 0.;
        q1 = node["q1"] ? node["q1"].as<int>() : 0;
        q3 = node["q3"] ? node["q3"].as<int>() : 0;
        ratio_mw2pp = node["ratio_mw2pp"] ? node["ratio_mw2pp"].as<double>() : 0.;
        resist_cu = node["resist_cu"] ? node["resist_cu"].as<double>() : 0.;
        sigma = node["sigma"] ? node["sigma"].as<double>() : 0.;
        y_tau_p = node["y_tau_p"] ? node["y_tau_p"].as<double>() : 0.;
        y_tau_pr = node["y_tau_pr"] ? node["y_tau_pr"].as<double>() : 0.;
        i_0 = node["i_0"] ? node["i_0"].as<double>() : 0.;
        d_r = node["d_r"] ? node["d_r"].as<double>() : 0.;
        h_m = node["h_m"] ? node["h_m"].as<double>() : 0.;
        h_0 = node["h_0"] ? node["h_0"].as<double>() : 0.;
        h_s = node["h_s"] ? node["h_s"].as<double>() : 0.;
        len_s = node["len_s"] ? node["len_s"].as<double>() : 0.;
        n_r = node["n_r"] ? node["n_r"].as<double>() : 0.;
        rad_ag = node["rad_ag"] ? node["rad_ag"].as<double>() : 0.;
        t_wr = node["t_wr"] ? node["t_wr"].as<double>() : 0.;
        n_s = node["n_s"] ? node["n_s"].as<double>() : 0.;
        b_st = node["b_st"] ? node["b_st"].as<double>() : 0.;
        d_s = node["d_s"] ? node["d_s"].as<double>() : 0.;
        t_ws = node["t_ws"] ? node["t_ws"].as<double>() : 0.;
        rho_copper = node["rho_copper"] ? node["rho_copper"].as<double>() : 0.;
        rho_fe = node["rho_fe"] ? node["rho_fe"].as<double>() : 0.;
        rho_fes = node["rho_fes"] ? node["rho_fes"].as<double>() : 0.;
        rho_pm = node["rho_pm"] ? node["rho_pm"].as<double>() : 0.;
        c_cu = node["c_cu"] ? node["c_cu"].as<double>() : 0.;
        c_fe = node["c_fe"] ? node["c_fe"].as<double>() : 0.;
        c_fes = node["c_fes"] ? node["c_fes"].as<double>() : 0.;
        c_pm = node["c_pm"] ? node["c_pm"].as<double>() : 0.;
        if (node["elastic_properties_mb"]) {
            elastic_properties_mb.parse(node["elastic_properties_mb"]);
        }
    }
};

// Drivetrain
struct Drivetrain {
    OuterShapeBem_2 outer_shape_bem; // Geometrical metrics describing the drivetrain. Currently, these are inputs to WISDEM specific drivetrain sizing tool, DrivetrainSE
    Gearbox gearbox; // Inputs describing the gearbox, when present
    Lss lss; // Inputs describing the low speed shaft
    Hss hss; // Inputs describing the high speed shaft, when present
    Nose nose; // Inputs describing the nose/turret at beginning (bedplate) and end (main bearing) points
    Bedplate bedplate; // Inputs describing the hollow elliptical bedplate used in direct drive configurations
    OtherComponents other_components; // Inputs describing all other drivetrain components, the assembly of brake, hvac, converter, transformer, and main bearings
    Generator generator;

    void parse(const YAML::Node& node) {
        if (node["outer_shape_bem"]) {
            outer_shape_bem.parse(node["outer_shape_bem"]);
        }
        if (node["gearbox"]) {
            gearbox.parse(node["gearbox"]);
        }
        if (node["lss"]) {
            lss.parse(node["lss"]);
        }
        if (node["hss"]) {
            hss.parse(node["hss"]);
        }
        if (node["nose"]) {
            nose.parse(node["nose"]);
        }
        if (node["bedplate"]) {
            bedplate.parse(node["bedplate"]);
        }
        if (node["other_components"]) {
            other_components.parse(node["other_components"]);
        }
        if (node["generator"]) {
            generator.parse(node["generator"]);
        }
    }
};

// Outer diameters of the tower defined from base (grid = 0) to top (grid = 1).
struct OuterDiameter {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Cd
struct Cd {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// OuterShapeBem_3
struct OuterShapeBem_3 {
    ReferenceAxis reference_axis;
    OuterDiameter outer_diameter; // Outer diameters of the tower defined from base (grid = 0) to top (grid = 1).
    Cd cd;

    void parse(const YAML::Node& node) {
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["outer_diameter"]) {
            outer_diameter.parse(node["outer_diameter"]);
        }
        if (node["cd"]) {
            cd.parse(node["cd"]);
        }
    }
};

// Layers_1
struct Layers_1 {
    std::string name; // structural component identifier
    std::string material; // material identifier
    Thickness thickness; // thickness of the laminate

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        material = node["material"] ? node["material"].as<std::string>() : "";
        if (node["thickness"]) {
            thickness.parse(node["thickness"]);
        }
    }
};

// InternalStructure2DFem_1
struct InternalStructure2DFem_1 {
    double outfitting_factor; // Multiplier of tower mass to account for the mass of the auxiliary systems, such as stairs, elevator, paint, or extra structural elements. This can be used to convert the mass of the steel cylinders to the total mass of the monopile.
    ReferenceAxis reference_axis;
    std::vector<Layers> layers; // ...

    void parse(const YAML::Node& node) {
        outfitting_factor = node["outfitting_factor"] ? node["outfitting_factor"].as<double>() : 0.;
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

// Data describing the wind turbine tower.
struct Tower {
    OuterShapeBem_3 outer_shape_bem;
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

// Added mass coefficient for the monopile defined in terms of grid and values that default to 1.
struct Ca {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// OuterShapeBem_4
struct OuterShapeBem_4 {
    ReferenceAxis reference_axis;
    OuterDiameter outer_diameter; // Outer diameters of the monopile defined from base (grid = 0) to top (grid = 1).
    Ca ca; // Added mass coefficient for the monopile defined in terms of grid and values that default to 1.
    Cd cd; // Drag coefficient for the monopile

    void parse(const YAML::Node& node) {
        if (node["reference_axis"]) {
            reference_axis.parse(node["reference_axis"]);
        }
        if (node["outer_diameter"]) {
            outer_diameter.parse(node["outer_diameter"]);
        }
        if (node["ca"]) {
            ca.parse(node["ca"]);
        }
        if (node["cd"]) {
            cd.parse(node["cd"]);
        }
    }
};

// Layers_2
struct Layers_2 {
    std::string name; // structural component identifier
    std::string material; // material identifier
    Thickness thickness; // thickness of the laminate

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        material = node["material"] ? node["material"].as<std::string>() : "";
        if (node["thickness"]) {
            thickness.parse(node["thickness"]);
        }
    }
};

// InternalStructure2DFem_2
struct InternalStructure2DFem_2 {
    double outfitting_factor; // Multiplier of monopile mass to account for the mass of the auxiliary systems, such as paint or extra structural elements. This can be used to convert the mass of the steel cylinders to the total mass of the monopile.
    ReferenceAxis reference_axis;
    std::vector<Layers> layers; // ...

    void parse(const YAML::Node& node) {
        outfitting_factor = node["outfitting_factor"] ? node["outfitting_factor"].as<double>() : 0.;
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
    double transition_piece_mass; // Total mass of transition piece
    double transition_piece_cost; // Total cost of transition piece
    double gravity_foundation_mass; // Total mass of gravity foundation addition onto monopile
    OuterShapeBem_4 outer_shape_bem;
    ElasticPropertiesMb elastic_properties_mb;
    InternalStructure2DFem_2 internal_structure_2d_fem;

    void parse(const YAML::Node& node) {
        transition_piece_mass = node["transition_piece_mass"] ? node["transition_piece_mass"].as<double>() : 0.;
        transition_piece_cost = node["transition_piece_cost"] ? node["transition_piece_cost"].as<double>() : 0.;
        gravity_foundation_mass = node["gravity_foundation_mass"] ? node["gravity_foundation_mass"].as<double>() : 0.;
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

// Jacket
struct Jacket {
    double transition_piece_mass; // Total mass of transition piece
    double transition_piece_cost; // Total cost of transition piece
    double gravity_foundation_mass; // Total mass of gravity foundation addition onto monopile
    std::string material; // Material of jacket members
    int n_bays; // Number of bays (x-joints) in the vertical direction for jackets.
    int n_legs; // Number of legs for jacket.
    double r_foot; // Radius of foot (bottom) of jacket, in meters.
    double r_head; // Radius of head (top) of jacket, in meters.
    double height; // Overall jacket height, meters.
    double leg_thickness; // Leg thickness, meters. Constant throughout each leg.
    std::vector<double> brace_diameters;
    std::vector<double> brace_thicknesses;
    std::vector<double> bay_spacing;
    std::vector<double> leg_spacing;
    bool x_mb; // Mud brace included if true.
    double leg_diameter; // Leg diameter, meters. Constant throughout each leg.

    void parse(const YAML::Node& node) {
        transition_piece_mass = node["transition_piece_mass"] ? node["transition_piece_mass"].as<double>() : 0.;
        transition_piece_cost = node["transition_piece_cost"] ? node["transition_piece_cost"].as<double>() : 0.;
        gravity_foundation_mass = node["gravity_foundation_mass"] ? node["gravity_foundation_mass"].as<double>() : 0.;
        material = node["material"] ? node["material"].as<std::string>() : "";
        n_bays = node["n_bays"] ? node["n_bays"].as<int>() : 0;
        n_legs = node["n_legs"] ? node["n_legs"].as<int>() : 0;
        r_foot = node["r_foot"] ? node["r_foot"].as<double>() : 0.;
        r_head = node["r_head"] ? node["r_head"].as<double>() : 0.;
        height = node["height"] ? node["height"].as<double>() : 0.;
        leg_thickness = node["leg_thickness"] ? node["leg_thickness"].as<double>() : 0.;
        brace_diameters = node["brace_diameters"] ? node["brace_diameters"].as<std::vector<double>>() : std::vector<double>();
        brace_thicknesses = node["brace_thicknesses"] ? node["brace_thicknesses"].as<std::vector<double>>() : std::vector<double>();
        bay_spacing = node["bay_spacing"] ? node["bay_spacing"].as<std::vector<double>>() : std::vector<double>();
        leg_spacing = node["leg_spacing"] ? node["leg_spacing"].as<std::vector<double>>() : std::vector<double>();
        x_mb = node["x_mb"] ? node["x_mb"].as<bool>() : false;
        leg_diameter = node["leg_diameter"] ? node["leg_diameter"].as<double>() : 0.;
    }
};

// If this joint is compliant is certain DOFs, then specify which are compliant (True) in the member/element coordinate system).  If not specified, default is all entries are False (completely rigid).  For instance, a ball joint would be Rx=Ry=Rz=False, Rxx=Ryy=Rzz=True
struct Reactions {
    bool rx;
    bool ry;
    bool rz;
    bool rxx;
    bool ryy;
    bool rzz;
    std::vector<double> euler; // Euler angles [alpha, beta, gamma] that describe the rotation of the Reaction coordinate system relative to the global coordinate system α is a rotation around the z axis, β is a rotation around the x' axis, γ is a rotation around the z" axis.

    void parse(const YAML::Node& node) {
        rx = node["rx"] ? node["rx"].as<bool>() : false;
        ry = node["ry"] ? node["ry"].as<bool>() : false;
        rz = node["rz"] ? node["rz"].as<bool>() : false;
        rxx = node["rxx"] ? node["rxx"].as<bool>() : false;
        ryy = node["ryy"] ? node["ryy"].as<bool>() : false;
        rzz = node["rzz"] ? node["rzz"].as<bool>() : false;
        euler = node["euler"] ? node["euler"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Joints
struct Joints {
    std::string name; // Unique name of the joint (node)
    std::vector<double> location; // Coordinates (x,y,z or r,θ,z) of the joint in the global coordinate system.
    bool transition; // Whether the transition piece and turbine tower attach at this node
    bool cylindrical; // Whether to use cylindrical coordinates (r,θ,z), with (r,θ) lying in the x/y-plane, instead of Cartesian coordinates.
    Reactions reactions; // If this joint is compliant is certain DOFs, then specify which are compliant (True) in the member/element coordinate system).  If not specified, default is all entries are False (completely rigid).  For instance, a ball joint would be Rx=Ry=Rz=False, Rxx=Ryy=Rzz=True

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        location = node["location"] ? node["location"].as<std::vector<double>>() : std::vector<double>();
        transition = node["transition"] ? node["transition"].as<bool>() : false;
        cylindrical = node["cylindrical"] ? node["cylindrical"].as<bool>() : false;
        if (node["reactions"]) {
            reactions.parse(node["reactions"]);
        }
    }
};

// OuterShape
struct OuterShape {
    std::string shape; // Specifies cross-sectional shape of the member.  If circular, then the outer_diameter field is required.  If polygonal, then the side_lengths, angles, and rotation fields are required
    OuterDiameter outer_diameter; // Gridded values describing diameter at non-dimensional axis from joint1 to joint2
    std::vector<double> side_lengths1; // Polygon side lengths at joint1
    std::vector<double> side_lengths2; // Polygon side lengths at joint1
    std::vector<double> angles; // Polygon angles with the ordering such that angle[i] is between side_length[i] and side_length[i+1]
    double rotation; // Angle between principle axes of the cross-section and the member coordinate system.  Essentially the rotation of the member if both joints were placed on the global x-y axis with the first side length along the z-axis

    void parse(const YAML::Node& node) {
        shape = node["shape"] ? node["shape"].as<std::string>() : "";
        if (node["outer_diameter"]) {
            outer_diameter.parse(node["outer_diameter"]);
        }
        side_lengths1 = node["side_lengths1"] ? node["side_lengths1"].as<std::vector<double>>() : std::vector<double>();
        side_lengths2 = node["side_lengths2"] ? node["side_lengths2"].as<std::vector<double>>() : std::vector<double>();
        angles = node["angles"] ? node["angles"].as<std::vector<double>>() : std::vector<double>();
        rotation = node["rotation"] ? node["rotation"].as<double>() : 0.;
    }
};

// Layers_3
struct Layers_3 {
    std::string name; // structural component identifier
    std::string material; // material identifier
    Thickness thickness; // Gridded values describing thickness along non-dimensional axis from joint1 to joint2

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        material = node["material"] ? node["material"].as<std::string>() : "";
        if (node["thickness"]) {
            thickness.parse(node["thickness"]);
        }
    }
};

// RingStiffeners
struct RingStiffeners {
    std::string material; // material identifier
    double flange_thickness;
    double flange_width;
    double web_height;
    double web_thickness;
    double spacing; // Spacing between stiffeners in non-dimensional grid coordinates. Value of 0.0 means no stiffeners

    void parse(const YAML::Node& node) {
        material = node["material"] ? node["material"].as<std::string>() : "";
        flange_thickness = node["flange_thickness"] ? node["flange_thickness"].as<double>() : 0.;
        flange_width = node["flange_width"] ? node["flange_width"].as<double>() : 0.;
        web_height = node["web_height"] ? node["web_height"].as<double>() : 0.;
        web_thickness = node["web_thickness"] ? node["web_thickness"].as<double>() : 0.;
        spacing = node["spacing"] ? node["spacing"].as<double>() : 0.;
    }
};

// LongitudinalStiffeners
struct LongitudinalStiffeners {
    std::string material; // material identifier
    double flange_thickness;
    double flange_width;
    double web_height;
    double web_thickness;
    double spacing; // Spacing between stiffeners in angle (deg). Value of 0.0 means no stiffeners

    void parse(const YAML::Node& node) {
        material = node["material"] ? node["material"].as<std::string>() : "";
        flange_thickness = node["flange_thickness"] ? node["flange_thickness"].as<double>() : 0.;
        flange_width = node["flange_width"] ? node["flange_width"].as<double>() : 0.;
        web_height = node["web_height"] ? node["web_height"].as<double>() : 0.;
        web_thickness = node["web_thickness"] ? node["web_thickness"].as<double>() : 0.;
        spacing = node["spacing"] ? node["spacing"].as<double>() : 0.;
    }
};

// Bulkhead
struct Bulkhead {
    std::string material; // material identifier
    Thickness thickness; // thickness of the bulkhead at non-dimensional locations of the member [0..1]

    void parse(const YAML::Node& node) {
        material = node["material"] ? node["material"].as<std::string>() : "";
        if (node["thickness"]) {
            thickness.parse(node["thickness"]);
        }
    }
};

// Ballast
struct Ballast {
    bool variable_flag; // If true, then this ballast is variable and adjusted by control system.  If false, then considered permanent
    std::string material; // material identifier
    std::vector<double> grid;
    double volume; // Total volume of ballast (permanent ballast only)

    void parse(const YAML::Node& node) {
        variable_flag = node["variable_flag"] ? node["variable_flag"].as<bool>() : false;
        material = node["material"] ? node["material"].as<std::string>() : "";
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        volume = node["volume"] ? node["volume"].as<double>() : 0.;
    }
};

// InternalStructure
struct InternalStructure {
    double outfitting_factor; // Scaling factor for the member mass to account for auxiliary structures, such as elevator, ladders, cables, platforms, fasteners, etc
    std::vector<Layers_3> layers; // Material layer properties
    RingStiffeners ring_stiffeners;
    LongitudinalStiffeners longitudinal_stiffeners;
    Bulkhead bulkhead;
    std::vector<Ballast> ballast; // Different types of permanent and/or variable ballast

    void parse(const YAML::Node& node) {
        outfitting_factor = node["outfitting_factor"] ? node["outfitting_factor"].as<double>() : 0.;
        if (node["layers"]) {
            for (const auto& item : node["layers"]) {
                Layers_3 x;
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
    std::string name; // Unique name of joint
    double grid; // Non-dimensional value along member axis

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        grid = node["grid"] ? node["grid"].as<double>() : 0.;
    }
};

// Members
struct Members {
    std::string name; // Name of the member
    std::string joint1; // Name of joint/node connection
    std::string joint2; // Name of joint/node connection
    OuterShape outer_shape;
    InternalStructure internal_structure;
    std::vector<AxialJoints> axial_joints; // Additional joints that are defined along the non-dimensional member axis can be defined here.  Unlike the joints defined in the global coordinate system in the :code:`joints` section of the ontology, these joints will change their absolute (x,y,z) location if the member diameter is altered the or :code:`joint1` or :code:`joint2` locations are changed during an optimization.  This is especially useful when designing a truss-like structure with pontoons attaching to primary members or columns in a semisubmersible. Like the joints above, these must be given a unique name.
    double ca; // User-defined added mass coefficient
    double cp; // User-defined pressure coefficient
    double cd; // User-defined drag coefficient

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        joint1 = node["joint1"] ? node["joint1"].as<std::string>() : "";
        joint2 = node["joint2"] ? node["joint2"].as<std::string>() : "";
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
        ca = node["ca"] ? node["ca"].as<double>() : 0.;
        cp = node["cp"] ? node["cp"].as<double>() : 0.;
        cd = node["cd"] ? node["cd"].as<double>() : 0.;
    }
};

// RigidBodies
struct RigidBodies {
    std::string joint1; // Name of joint/node connection
    double mass; // Mass of this rigid body
    double cost; // Cost of this rigid body
    std::vector<double> cm_offset; // Offset from joint location to center of mass (CM) of body in dx, dy, dz
    std::vector<double> moments_of_inertia; // Moments of inertia around body CM in Ixx, Iyy, Izz
    double ca; // User-defined added mass coefficient
    double cp; // User-defined pressure coefficient
    double cd; // User-defined drag coefficient

    void parse(const YAML::Node& node) {
        joint1 = node["joint1"] ? node["joint1"].as<std::string>() : "";
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        cost = node["cost"] ? node["cost"].as<double>() : 0.;
        cm_offset = node["cm_offset"] ? node["cm_offset"].as<std::vector<double>>() : std::vector<double>();
        moments_of_inertia = node["moments_of_inertia"] ? node["moments_of_inertia"].as<std::vector<double>>() : std::vector<double>();
        ca = node["ca"] ? node["ca"].as<double>() : 0.;
        cp = node["cp"] ? node["cp"].as<double>() : 0.;
        cd = node["cd"] ? node["cd"].as<double>() : 0.;
    }
};

// The floating platform ontology uses a *graph*-like representation of the geometry with Joints and Members.  Additional rigid body point masses can be defined at the joints as well.
struct FloatingPlatform {
    std::vector<Joints> joints; // Joints are the *nodes* of the graph representation of the floating platform.  They must be assigned a unique name for later reference by the members.
    std::vector<Members> members;
    std::vector<RigidBodies> rigid_bodies; // Additional point masses at joints with user-customized properties
    double transition_piece_mass; // Total mass of transition piece
    double transition_piece_cost; // Total cost of transition piece

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
        transition_piece_mass = node["transition_piece_mass"] ? node["transition_piece_mass"].as<double>() : 0.;
        transition_piece_cost = node["transition_piece_cost"] ? node["transition_piece_cost"].as<double>() : 0.;
    }
};

// Nodes
struct Nodes {
    std::string name; // Name or ID of this node for use in line segment
    std::string node_type;
    std::vector<double> location; // – Coordinates x, y, and z of the connection (relative to inertial reference frame if Fixed or Connect, relative to platform reference frame if Vessel). In the case of Connect nodes, it is simply an initial guess for position before MoorDyn calculates the equilibrium initial position.
    std::string joint; // For anchor positions and fairlead attachments, reference a joint name from the "joints" section or an "axial_joint" on a member
    std::string anchor_type; // Name of anchor type from anchor_type list
    std::string fairlead_type;
    double node_mass; // Clump weight mass
    double node_volume; // Floater volume
    double drag_area; // Product of drag coefficient and projected area (assumed constant in all directions) to calculate a drag force for the node
    double added_mass; // Added mass coefficient used along with node volume to calculate added mass on node

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        node_type = node["node_type"] ? node["node_type"].as<std::string>() : "";
        location = node["location"] ? node["location"].as<std::vector<double>>() : std::vector<double>();
        joint = node["joint"] ? node["joint"].as<std::string>() : "";
        anchor_type = node["anchor_type"] ? node["anchor_type"].as<std::string>() : "";
        fairlead_type = node["fairlead_type"] ? node["fairlead_type"].as<std::string>() : "";
        node_mass = node["node_mass"] ? node["node_mass"].as<double>() : 0.;
        node_volume = node["node_volume"] ? node["node_volume"].as<double>() : 0.;
        drag_area = node["drag_area"] ? node["drag_area"].as<double>() : 0.;
        added_mass = node["added_mass"] ? node["added_mass"].as<double>() : 0.;
    }
};

// Lines
struct Lines {
    std::string name; // ID of this line
    std::string line_type; // Reference to line type database
    double unstretched_length; // length of line segment prior to tensioning
    std::string node1; // node id of first line connection
    std::string node2; // node id of second line connection

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        line_type = node["line_type"] ? node["line_type"].as<std::string>() : "";
        unstretched_length = node["unstretched_length"] ? node["unstretched_length"].as<double>() : 0.;
        node1 = node["node1"] ? node["node1"].as<std::string>() : "";
        node2 = node["node2"] ? node["node2"].as<std::string>() : "";
    }
};

// LineTypes
struct LineTypes {
    std::string name; // Name of material or line type to be referenced by line segments
    double diameter; // the volume-equivalent diameter of the line – the diameter of a cylinder having the same displacement per unit length
    std::string type; // Type of material for property lookup
    double mass_density; // mass per unit length (in air)
    double stiffness; // axial line stiffness, product of elasticity modulus and cross-sectional area
    double cost; // cost per unit length
    double breaking_load; // line break tension
    double damping; // internal damping (BA)
    double transverse_added_mass; // transverse added mass coefficient (with respect to line displacement)
    double tangential_added_mass; // tangential added mass coefficient (with respect to line displacement)
    double transverse_drag; // transverse drag coefficient (with respect to frontal area, d*l)
    double tangential_drag; // tangential drag coefficient (with respect to surface area, π*d*l)

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        diameter = node["diameter"] ? node["diameter"].as<double>() : 0.;
        type = node["type"] ? node["type"].as<std::string>() : "";
        mass_density = node["mass_density"] ? node["mass_density"].as<double>() : 0.;
        stiffness = node["stiffness"] ? node["stiffness"].as<double>() : 0.;
        cost = node["cost"] ? node["cost"].as<double>() : 0.;
        breaking_load = node["breaking_load"] ? node["breaking_load"].as<double>() : 0.;
        damping = node["damping"] ? node["damping"].as<double>() : 0.;
        transverse_added_mass = node["transverse_added_mass"] ? node["transverse_added_mass"].as<double>() : 0.;
        tangential_added_mass = node["tangential_added_mass"] ? node["tangential_added_mass"].as<double>() : 0.;
        transverse_drag = node["transverse_drag"] ? node["transverse_drag"].as<double>() : 0.;
        tangential_drag = node["tangential_drag"] ? node["tangential_drag"].as<double>() : 0.;
    }
};

// AnchorTypes
struct AnchorTypes {
    std::string name; // Name of anchor to be referenced by anchor_id in Nodes section
    std::string type; // Type of anchor for property lookup
    double mass; // mass of the anchor
    double cost; // cost of the anchor
    double max_lateral_load; // Maximum lateral load (parallel to the sea floor) that the anchor can support
    double max_vertical_load; // Maximum vertical load (perpendicular to the sea floor) that the anchor can support

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        type = node["type"] ? node["type"].as<std::string>() : "";
        mass = node["mass"] ? node["mass"].as<double>() : 0.;
        cost = node["cost"] ? node["cost"].as<double>() : 0.;
        max_lateral_load = node["max_lateral_load"] ? node["max_lateral_load"].as<double>() : 0.;
        max_vertical_load = node["max_vertical_load"] ? node["max_vertical_load"].as<double>() : 0.;
    }
};

// The mooring system ontology follows closely the input file format for MoorDyn or MAP++.  Additional information can be found in the `MoorDyn user guide <http://www.matt-hall.ca/files/MoorDyn-Users-Guide-2017-08-16.pdf>`_ .
struct Mooring {
    std::vector<Nodes> nodes; // List of nodes in the mooring system
    std::vector<Lines> lines; // List of all mooring line properties in the mooring system
    std::vector<LineTypes> line_types; // List of mooring line properties used in the system
    std::vector<AnchorTypes> anchor_types; // List of anchor properties used in the system

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

// Nested dictionary structure of components describing the wind turbine assembly
struct Components {
    Blade blade; // The component :code:`blade` includes three subcomponents, namely :code:`outer_shape_bem`, :code:`elastic_properties_mb`, and :code:`internal_structure_2d_fem`. A fourth and a fifth subfields :code:`cfd_geometry` and :code:`3D_fem` will be added to support higher fidelity modeling of the rotor. All distributed quantities, such as blade chord or the thickness of a structural component, are expressed in terms of pair arrays :code:`grid` and :code:`values`, which must have a minimum length of two elements and the same size. :code:`grid` is defined nondimensional between 0 (root) and 1 (tip) along the, possibly curved, :code:`reference_axis`.
    Hub hub;
    Drivetrain drivetrain;
    Tower tower; // Data describing the wind turbine tower.
    Monopile monopile;
    Jacket jacket;
    FloatingPlatform floating_platform; // The floating platform ontology uses a *graph*-like representation of the geometry with Joints and Members.  Additional rigid body point masses can be defined at the joints as well.
    Mooring mooring; // The mooring system ontology follows closely the input file format for MoorDyn or MAP++.  Additional information can be found in the `MoorDyn user guide <http://www.matt-hall.ca/files/MoorDyn-Users-Guide-2017-08-16.pdf>`_ .

    void parse(const YAML::Node& node) {
        if (node["blade"]) {
            blade.parse(node["blade"]);
        }
        if (node["hub"]) {
            hub.parse(node["hub"]);
        }
        if (node["drivetrain"]) {
            drivetrain.parse(node["drivetrain"]);
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

// The airfoil :code:`coordinates` are specified here in the sub-fields :code:`x` and :code:`y`. :code:`x` and :code:`y` must have the same length. :code:`x` must be defined between 0, which corresponds to the leading edge, and 1, which corresponds to the trailing edge. Airfoil coordinates should be defined from trailing edge (:code:`x=1`) towards the suction side (mostly positive :code:`y` values), to leading edge (:code:`x=0`, :code:`y=0`), to the pressure side (mostly negative :code:`y`), and conclude at the trailing edge pressure side (:code:`x=1`). Flatback airfoils can be defined either open (:code:`x=1`, :code:`y!=0`) or closed (:code:`x=1`, :code:`y=0`).
struct Coordinates {
    std::vector<double> x;
    std::vector<double> y;

    void parse(const YAML::Node& node) {
        x = node["x"] ? node["x"].as<std::vector<double>>() : std::vector<double>();
        y = node["y"] ? node["y"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Lift coefficient as a function of angle of attack (deg)
struct Cl {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Moment coefficient as a function of angle of attack (deg)
struct Cm {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// List of one or more sets of polars for the airfoil. The field :code:`polars` must include the sub-fields :code:`configuration`, :code:`re`, :code:`cl`, :code:`cd`, and :code:`cm`. Parameters characterizing the unsteady aerodynamics are optional.
struct Polars {
    std::string configuration; // Text to identify the setup for the definition of the polars
    double re; // Reynolds number of the polars
    Cl cl; // Lift coefficient as a function of angle of attack (deg)
    Cd cd; // Drag coefficient as a function of angle of attack (deg)
    Cm cm; // Moment coefficient as a function of angle of attack (deg)
    double aoa0; // 0-lift angle of attack
    double aoa1; // Angle of attack at f=0.7, (approximately the stall angle) for AOA>alpha0
    double aoa2; // Angle of attack at f=0.7, (approximately the stall angle) for AOA<alpha0
    double eta_e; // Recovery factor in the range [0.85 - 0.95]
    double c_nalpha; // Slope of the 2D normal force coefficient curve
    double t_f0; // Initial value of the time constant associated with Df in the expression of Df and f. [default = 3]
    double t_v0; // Initial value of the time constant associated with the vortex lift decay process; it is used in the expression of Cvn. It depends on Re,M, and airfoil class. [default = 6]
    double t_p; // Boundary-layer,leading edge pressure gradient time constant in the expression of Dp. It should be tuned based on airfoil experimental data. [default = 1.7]
    double t_vl; // Initial value of the time constant associated with the vortex advection process; it represents the non-dimensional time in semi-chords, needed for a vortex to travel from LE to trailing edge (TE); it is used in the expression of Cvn. It depends on Re, M (weakly), and airfoil. [valid range = 6 - 13, default = 11]
    double b1; // Constant in the expression of phi_alpha^c and phi_q^c.  This value is relatively insensitive for thin airfoils, but may be different for turbine airfoils. [from experimental results, defaults to 0.14]
    double b2; // Constant in the expression of phi_alpha^c and phi_q^c.  This value is relatively insensitive for thin airfoils, but may be different for turbine airfoils. [from experimental results, defaults to 0.53]
    double b5; // Constant in the expression of K'''_q,cm_q^nc, and k_m,q.  [from  experimental results, defaults to 5]
    double a1; // Constant in the expression of phi_alpha^c and phi_q^c.  This value is relatively insensitive for thin airfoils, but may be different for turbine airfoils. [from experimental results, defaults to 0.3]
    double a2; // Constant in the expression of phi_alpha^c and phi_q^c.  This value is relatively insensitive for thin airfoils, but may be different for turbine airfoils. [from experimental results, defaults to 0.7]
    double a5; // Constant in the expression of K'''_q,cm_q^nc, and k_m,q. [from experimental results, defaults to 1]
    double s1; // Constant in the f curve best-fit for alpha0<=AOA<=alpha1; by definition it depends on the airfoil.
    double s2; // Constant in the f curve best-fit for AOA> alpha1; by definition it depends on the airfoil.
    double s3; // Constant in the f curve best-fit for alpha2<=AOA< alpha0; by definition it depends on the airfoil.
    double s4; // Constant in the f curve best-fit for AOA< alpha2; by definition it depends on the airfoil.
    double cn1; // Critical value of C0n at leading edge separation. It should be extracted from airfoil data at a given Mach and Reynolds number. It can be calculated from the static value of Cn at either the break in the pitching moment or the loss of chord force at the onset of stall. It is close to the condition of maximum lift of the airfoil at low Mach numbers.
    double cn2; // As Cn1 for negative AOAs
    double st_sh; // Strouhal's shedding frequency constant. [default = 0.19]
    double cd0; // 2D drag coefficient value at 0-lift.
    double cm0; // 2D pitching moment coefficient about 1/4-chord location, at 0-lift, positive if nose up.
    double k0; // Constant in the \hat(x)_cp curve best-fit; = (\hat(x)_AC-0.25).
    double k1; // Constant in the \hat(x)_cp curve best-fit.
    double k2; // Constant in the \hat(x)_cp curve best-fit.
    double k3; // Constant in the \hat(x)_cp curve best-fit.
    double k1_hat; // Constant in the expression of Cc due to leading edge vortex effects.
    double x_cp_bar; // Constant in the expression of \hat(x)_cp^v. [default = 0.2]
    double uacutout; // Angle of attack above which unsteady aerodynamics are disabled (deg). [Specifying the string "Default" sets UACutout to 45 degrees]
    double filtcutoff; // Cut-off frequency (-3 dB corner frequency) for low-pass filtering the AoA input to UA, as well as the 1st and 2nd derivatives (Hz) [default = 20]

    void parse(const YAML::Node& node) {
        configuration = node["configuration"] ? node["configuration"].as<std::string>() : "";
        re = node["re"] ? node["re"].as<double>() : 0.;
        if (node["cl"]) {
            cl.parse(node["cl"]);
        }
        if (node["cd"]) {
            cd.parse(node["cd"]);
        }
        if (node["cm"]) {
            cm.parse(node["cm"]);
        }
        aoa0 = node["aoa0"] ? node["aoa0"].as<double>() : 0.;
        aoa1 = node["aoa1"] ? node["aoa1"].as<double>() : 0.;
        aoa2 = node["aoa2"] ? node["aoa2"].as<double>() : 0.;
        eta_e = node["eta_e"] ? node["eta_e"].as<double>() : 0.;
        c_nalpha = node["c_nalpha"] ? node["c_nalpha"].as<double>() : 0.;
        t_f0 = node["t_f0"] ? node["t_f0"].as<double>() : 0.;
        t_v0 = node["t_v0"] ? node["t_v0"].as<double>() : 0.;
        t_p = node["t_p"] ? node["t_p"].as<double>() : 0.;
        t_vl = node["t_vl"] ? node["t_vl"].as<double>() : 0.;
        b1 = node["b1"] ? node["b1"].as<double>() : 0.;
        b2 = node["b2"] ? node["b2"].as<double>() : 0.;
        b5 = node["b5"] ? node["b5"].as<double>() : 0.;
        a1 = node["a1"] ? node["a1"].as<double>() : 0.;
        a2 = node["a2"] ? node["a2"].as<double>() : 0.;
        a5 = node["a5"] ? node["a5"].as<double>() : 0.;
        s1 = node["s1"] ? node["s1"].as<double>() : 0.;
        s2 = node["s2"] ? node["s2"].as<double>() : 0.;
        s3 = node["s3"] ? node["s3"].as<double>() : 0.;
        s4 = node["s4"] ? node["s4"].as<double>() : 0.;
        cn1 = node["cn1"] ? node["cn1"].as<double>() : 0.;
        cn2 = node["cn2"] ? node["cn2"].as<double>() : 0.;
        st_sh = node["st_sh"] ? node["st_sh"].as<double>() : 0.;
        cd0 = node["cd0"] ? node["cd0"].as<double>() : 0.;
        cm0 = node["cm0"] ? node["cm0"].as<double>() : 0.;
        k0 = node["k0"] ? node["k0"].as<double>() : 0.;
        k1 = node["k1"] ? node["k1"].as<double>() : 0.;
        k2 = node["k2"] ? node["k2"].as<double>() : 0.;
        k3 = node["k3"] ? node["k3"].as<double>() : 0.;
        k1_hat = node["k1_hat"] ? node["k1_hat"].as<double>() : 0.;
        x_cp_bar = node["x_cp_bar"] ? node["x_cp_bar"].as<double>() : 0.;
        uacutout = node["uacutout"] ? node["uacutout"].as<double>() : 0.;
        filtcutoff = node["filtcutoff"] ? node["filtcutoff"].as<double>() : 0.;
    }
};

// Airfoils
struct Airfoils {
    std::string name; // Name of the airfoil
    Coordinates coordinates; // The airfoil :code:`coordinates` are specified here in the sub-fields :code:`x` and :code:`y`. :code:`x` and :code:`y` must have the same length. :code:`x` must be defined between 0, which corresponds to the leading edge, and 1, which corresponds to the trailing edge. Airfoil coordinates should be defined from trailing edge (:code:`x=1`) towards the suction side (mostly positive :code:`y` values), to leading edge (:code:`x=0`, :code:`y=0`), to the pressure side (mostly negative :code:`y`), and conclude at the trailing edge pressure side (:code:`x=1`). Flatback airfoils can be defined either open (:code:`x=1`, :code:`y!=0`) or closed (:code:`x=1`, :code:`y=0`).
    double relative_thickness; // Float defined between 0 (plate) and 1 (cylinder) to specify the relative thickness of the airfoil. This generates a small redundancy (airfoil thickness could be determined from the field coordinates), but it simplifies the converters.
    double aerodynamic_center; // Float defined between 0 (leading edge) and 1 (trailing edge) to specify the chordwise coordinate of the aerodynamic center used to define the polars.
    std::vector<Polars> polars; // Different sets of polars at varying conditions.

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        if (node["coordinates"]) {
            coordinates.parse(node["coordinates"]);
        }
        relative_thickness = node["relative_thickness"] ? node["relative_thickness"].as<double>() : 0.;
        aerodynamic_center = node["aerodynamic_center"] ? node["aerodynamic_center"].as<double>() : 0.;
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
    std::string name; // Name of the material
    std::string description; // Optional string to describe the origin of the material, for example referencing a report or a paper
    std::string source; // Optional field describing where the data come from
    int orth; // Flag specifying whether a material is isotropic (0) or orthotropic (1). This determines whether some of the fields below are specified as a float or an array of floats.
    double rho; // Density of the material. For composites, this is the density of the laminate once cured.
    std::variant<double, std::vector<double>> e; // Stiffness modulus. For orthotropic materials, it consists of an array with E11, E22, and E33.
    std::variant<double, std::vector<double>> g; // Shear stiffness modulus. For orthotropic materials, it consists of an array with G12, G13, and G23.
    std::variant<double, std::vector<double>> nu; // Poisson ratio. For orthotropic materials, it consists of an array with nu12, nu13 and nu23. For isotropic materials, a minimum of -1 and a maximum of 0.5 are imposed. No limits are imposed to anisotropic materials.
    std::variant<double, std::vector<double>> alpha; // Thermal coefficient of expansion. For orthotropic materials, it consists of an array with alpha11, alpha22, and alpha33.
    std::variant<double, std::vector<double>> xt; // Ultimate tensile strength. For orthotropic materials, it consists of an array with Xt11, Xt22, and Xt33.
    std::variant<double, std::vector<double>> xc; // Ultimate compressive strength. For orthotropic materials, it consists of an array with Xc11, Xc22, and Xc33. Values are defined positive.
    std::variant<double, std::vector<double>> xy; // Ultimate yield strength for metals. For orthotropic materials, it consists of an array with the strength in directions 12, 13 and 23
    std::variant<double, std::vector<double>> s; // Ultimate shear strength. For orthotropic materials, it consists of an array with the strength in directions 12, 13 and 23. Values are defined positive.
    double ply_t; // Ply thickness of a composite material. The unit of measure is m. The actual laminate thickness is defined in the fields :code:`components`.
    double unit_cost; // Unit cost of the material. For composites, this is the unit cost of the dry fabric.
    double fvf; // Fiber volume fraction of a composite material. The minimum values is 0 (only matrix), the maximum value is 1 (only fibers).
    double fwf; // Fiber weight fraction of a composite material. The minimum values is 0 (only matrix), the maximum value is 1 (only fibers).
    double fiber_density; // Density of the fibers of a composite material. Standard glass fiber has a fiber density of approximately 2600 kg/m3, while standard carbon fiber has a fiber density of approximately 1800 kg/m3.
    double area_density_dry; // Aerial density of a fabric of a composite material.
    int manufacturing_id; // Flag to define the manufacturing process behind the laminate, for example 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
    double waste; // Fraction of material that ends up wasted during manufacturing.
    double roll_mass; // Mass of a fabric roll.
    double gic; // Mode 1 critical energy-release rate.
    double giic; // Mode 2 critical energy-release rate.
    double alp0; // Fracture angle under pure transverse compression.
    std::variant<double, std::vector<double>> a; // Fatigue S/N curve fitting parameter S=A*N^(-1/m).
    std::variant<double, std::vector<double>> m; // Fatigue S/N curve fitting parameter S=A*N^(-1/m)
    std::variant<double, std::vector<double>> r; // Fatigue stress ratio

    void parse(const YAML::Node& node) {
        name = node["name"] ? node["name"].as<std::string>() : "";
        description = node["description"] ? node["description"].as<std::string>() : "";
        source = node["source"] ? node["source"].as<std::string>() : "";
        orth = node["orth"] ? node["orth"].as<int>() : 0;
        rho = node["rho"] ? node["rho"].as<double>() : 0.;
        e = node["e"] ? node["e"].as<double>() : 0.;
        g = node["g"] ? node["g"].as<double>() : 0.;        if (!orth) {    
            nu = node["nu"] ? node["nu"].as<double>() : 0.;
        } else {    
            nu = node["nu"] ? node["nu"].as<std::vector<double>>() : std::vector<double>();
        }        if (!orth) {    
            alpha = node["alpha"] ? node["alpha"].as<double>() : 0.;
        } else {    
            alpha = node["alpha"] ? node["alpha"].as<std::vector<double>>() : std::vector<double>();
        }
        xt = node["xt"] ? node["xt"].as<double>() : 0.;
        xc = node["xc"] ? node["xc"].as<double>() : 0.;
        xy = node["xy"] ? node["xy"].as<double>() : 0.;
        s = node["s"] ? node["s"].as<double>() : 0.;
        ply_t = node["ply_t"] ? node["ply_t"].as<double>() : 0.;
        unit_cost = node["unit_cost"] ? node["unit_cost"].as<double>() : 0.;
        fvf = node["fvf"] ? node["fvf"].as<double>() : 0.;
        fwf = node["fwf"] ? node["fwf"].as<double>() : 0.;
        fiber_density = node["fiber_density"] ? node["fiber_density"].as<double>() : 0.;
        area_density_dry = node["area_density_dry"] ? node["area_density_dry"].as<double>() : 0.;
        manufacturing_id = node["manufacturing_id"] ? node["manufacturing_id"].as<int>() : 0;
        waste = node["waste"] ? node["waste"].as<double>() : 0.;
        roll_mass = node["roll_mass"] ? node["roll_mass"].as<double>() : 0.;
        gic = node["gic"] ? node["gic"].as<double>() : 0.;
        giic = node["giic"] ? node["giic"].as<double>() : 0.;
        alp0 = node["alp0"] ? node["alp0"].as<double>() : 0.;
        a = node["a"] ? node["a"].as<double>() : 0.;
        m = node["m"] ? node["m"].as<double>() : 0.;
        r = node["r"] ? node["r"].as<double>() : 0.;
    }
};

// Supervisory
struct Supervisory {
    double vin; // Cut-in wind speed of the wind turbine.
    double vout; // Cut-out wind speed of the wind turbine.
    double maxts; // Maximum allowable blade tip speed.

    void parse(const YAML::Node& node) {
        vin = node["vin"] ? node["vin"].as<double>() : 0.;
        vout = node["vout"] ? node["vout"].as<double>() : 0.;
        maxts = node["maxts"] ? node["maxts"].as<double>() : 0.;
    }
};

// Pitch
struct Pitch {
    double min_pitch; // Minimum pitch angle, where the default is 0 deg
    double max_pitch; // Maximum pitch angle, where the default is 90 deg
    double max_pitch_rate; // Maximum pitch rate of the rotor blades.

    void parse(const YAML::Node& node) {
        min_pitch = node["min_pitch"] ? node["min_pitch"].as<double>() : 0.;
        max_pitch = node["max_pitch"] ? node["max_pitch"].as<double>() : 0.;
        max_pitch_rate = node["max_pitch_rate"] ? node["max_pitch_rate"].as<double>() : 0.;
    }
};

// Torque
struct Torque {
    double max_torque_rate; // Maximum torque rate of the wind turbine generator.
    double tsr; // Rated tip speed ratio of the wind turbine. As default, it is maintained constant in region II.
    double vs_minspd; // Minimum rotor speed.
    double vs_maxspd; // Maximum rotor speed.

    void parse(const YAML::Node& node) {
        max_torque_rate = node["max_torque_rate"] ? node["max_torque_rate"].as<double>() : 0.;
        tsr = node["tsr"] ? node["tsr"].as<double>() : 0.;
        vs_minspd = node["vs_minspd"] ? node["vs_minspd"].as<double>() : 0.;
        vs_maxspd = node["vs_maxspd"] ? node["vs_maxspd"].as<double>() : 0.;
    }
};

// Dictionary reporting the data describing the wind turbine controller
struct Control {
    Supervisory supervisory;
    Pitch pitch;
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

// The field :code:`environment` includes the data characterizing air and water where the wind turbine operates.
struct Environment {
    double gravity; // Gravitational acceleration
    double air_density; // Density of air.
    double air_dyn_viscosity; // Dynamic viscosity of air.
    double air_pressure; // Atmospheric pressure of air
    double air_vapor_pressure; // Vapor pressure of fluid
    double weib_shape_parameter; // Shape factor of the Weibull wind distribution.
    double air_speed_sound; // Speed of sound in air.
    double shear_exp; // Shear exponent of the atmospheric boundary layer.
    double water_density; // Density of water.
    double water_dyn_viscosity; // Dynamic viscosity of water.
    double water_depth; // Water depth for offshore environment.
    double soil_shear_modulus; // Shear modulus of the soil.
    double soil_poisson; // Poisson ratio of the soil.
    double v_mean; // Average inflow wind speed. If different than 0, this will overwrite the V mean of the IEC wind class

    void parse(const YAML::Node& node) {
        gravity = node["gravity"] ? node["gravity"].as<double>() : 0.;
        air_density = node["air_density"] ? node["air_density"].as<double>() : 0.;
        air_dyn_viscosity = node["air_dyn_viscosity"] ? node["air_dyn_viscosity"].as<double>() : 0.;
        air_pressure = node["air_pressure"] ? node["air_pressure"].as<double>() : 0.;
        air_vapor_pressure = node["air_vapor_pressure"] ? node["air_vapor_pressure"].as<double>() : 0.;
        weib_shape_parameter = node["weib_shape_parameter"] ? node["weib_shape_parameter"].as<double>() : 0.;
        air_speed_sound = node["air_speed_sound"] ? node["air_speed_sound"].as<double>() : 0.;
        shear_exp = node["shear_exp"] ? node["shear_exp"].as<double>() : 0.;
        water_density = node["water_density"] ? node["water_density"].as<double>() : 0.;
        water_dyn_viscosity = node["water_dyn_viscosity"] ? node["water_dyn_viscosity"].as<double>() : 0.;
        water_depth = node["water_depth"] ? node["water_depth"].as<double>() : 0.;
        soil_shear_modulus = node["soil_shear_modulus"] ? node["soil_shear_modulus"].as<double>() : 0.;
        soil_poisson = node["soil_poisson"] ? node["soil_poisson"].as<double>() : 0.;
        v_mean = node["v_mean"] ? node["v_mean"].as<double>() : 0.;
    }
};

// Data for a balance of station cost analysis.
struct Bos {
    double plant_turbine_spacing; // Distance between turbines in the primary grid streamwise direction in rotor diameters
    double plant_row_spacing; // Distance between turbine rows in the cross-wind direction in rotor diameters
    double commissioning_pct; // Fraction of total BOS cost that is due to commissioning
    double decommissioning_pct; // Fraction of total BOS cost that is due to decommissioning
    double distance_to_substation; // Distance from centroid of plant to substation in km
    double distance_to_interconnection; // Distance from substation to grid connection in km
    double distance_to_landfall; // Distance from plant centroid to export cable landfall for offshore plants
    double distance_to_site; // Distance from port to plant centroid for offshore plants
    double interconnect_voltage; // Voltage of cabling to grid interconnection
    double port_cost_per_month; // Monthly port rental fees
    double site_auction_price; // Cost to secure site lease
    double site_assessment_plan_cost; // Cost to do engineering plan for site assessment
    double site_assessment_cost; // Cost to execute site assessment
    double construction_operations_plan_cost; // Cost to do construction planning
    double boem_review_cost; // Cost for additional review by U.S. Dept of Interior Bureau of Ocean Energy Management (BOEM)
    double design_install_plan_cost; // Cost to do installation planning

    void parse(const YAML::Node& node) {
        plant_turbine_spacing = node["plant_turbine_spacing"] ? node["plant_turbine_spacing"].as<double>() : 0.;
        plant_row_spacing = node["plant_row_spacing"] ? node["plant_row_spacing"].as<double>() : 0.;
        commissioning_pct = node["commissioning_pct"] ? node["commissioning_pct"].as<double>() : 0.;
        decommissioning_pct = node["decommissioning_pct"] ? node["decommissioning_pct"].as<double>() : 0.;
        distance_to_substation = node["distance_to_substation"] ? node["distance_to_substation"].as<double>() : 0.;
        distance_to_interconnection = node["distance_to_interconnection"] ? node["distance_to_interconnection"].as<double>() : 0.;
        distance_to_landfall = node["distance_to_landfall"] ? node["distance_to_landfall"].as<double>() : 0.;
        distance_to_site = node["distance_to_site"] ? node["distance_to_site"].as<double>() : 0.;
        interconnect_voltage = node["interconnect_voltage"] ? node["interconnect_voltage"].as<double>() : 0.;
        port_cost_per_month = node["port_cost_per_month"] ? node["port_cost_per_month"].as<double>() : 0.;
        site_auction_price = node["site_auction_price"] ? node["site_auction_price"].as<double>() : 0.;
        site_assessment_plan_cost = node["site_assessment_plan_cost"] ? node["site_assessment_plan_cost"].as<double>() : 0.;
        site_assessment_cost = node["site_assessment_cost"] ? node["site_assessment_cost"].as<double>() : 0.;
        construction_operations_plan_cost = node["construction_operations_plan_cost"] ? node["construction_operations_plan_cost"].as<double>() : 0.;
        boem_review_cost = node["boem_review_cost"] ? node["boem_review_cost"].as<double>() : 0.;
        design_install_plan_cost = node["design_install_plan_cost"] ? node["design_install_plan_cost"].as<double>() : 0.;
    }
};

// Data for a levelized cost of energy analysis.
struct Costs {
    double wake_loss_factor; // Factor to model losses in annual energy production in a wind farm compared to the annual energy production at the turbine level (wakes mostly).
    double fixed_charge_rate; // Fixed charge rate to compute the levelized cost of energy. See this for inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    double bos_per_kw; // Balance of stations costs expressed in USD per kW. See this for inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    double opex_per_kw; // Operational expenditures expressed in USD per kW. See this for inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    int turbine_number; // Number of turbines in the park, used to compute levelized cost of energy. Often wind parks are assumed of 600 MW. See this for inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf
    double labor_rate; // Hourly loaded wage per worker including all benefits and overhead.  This is currently only applied to steel, column structures.
    double painting_rate; // Cost per unit area for finishing and surface treatments.  This is currently only applied to steel, column structures.
    double blade_mass_cost_coeff; // Regression-based blade cost/mass ratio
    double hub_mass_cost_coeff; // Regression-based hub cost/mass ratio
    double pitch_system_mass_cost_coeff; // Regression-based pitch system cost/mass ratio
    double spinner_mass_cost_coeff; // Regression-based spinner cost/mass ratio
    double lss_mass_cost_coeff; // Regression-based low speed shaft cost/mass ratio
    double bearing_mass_cost_coeff; // Regression-based bearing cost/mass ratio
    double gearbox_mass_cost_coeff; // Regression-based gearbox cost/mass ratio
    double hss_mass_cost_coeff; // Regression-based high speed side cost/mass ratio
    double generator_mass_cost_coeff; // Regression-based generator cost/mass ratio
    double bedplate_mass_cost_coeff; // Regression-based bedplate cost/mass ratio
    double yaw_mass_cost_coeff; // Regression-based yaw system cost/mass ratio
    double converter_mass_cost_coeff; // Regression-based converter cost/mass ratio
    double transformer_mass_cost_coeff; // Regression-based transformer cost/mass ratio
    double hvac_mass_cost_coeff; // Regression-based HVAC system cost/mass ratio
    double cover_mass_cost_coeff; // Regression-based nacelle cover cost/mass ratio
    double elec_connec_machine_rating_cost_coeff; // Regression-based electrical plant connection cost/rating ratio
    double platforms_mass_cost_coeff; // Regression-based nacelle platform cost/mass ratio
    double tower_mass_cost_coeff; // Regression-based tower cost/mass ratio
    double controls_machine_rating_cost_coeff; // Regression-based controller and sensor system cost/rating ratio
    double crane_cost; // crane cost if present
    double electricity_price; // Electricity price used to compute value in beyond lcoe metrics
    double reserve_margin_price; // Reserve margin price used to compute value in beyond lcoe metrics
    double capacity_credit; // Capacity credit used to compute value in beyond lcoe metrics
    double benchmark_price; // Benchmark price used to nondimensionalize value in beyond lcoe metrics

    void parse(const YAML::Node& node) {
        wake_loss_factor = node["wake_loss_factor"] ? node["wake_loss_factor"].as<double>() : 0.;
        fixed_charge_rate = node["fixed_charge_rate"] ? node["fixed_charge_rate"].as<double>() : 0.;
        bos_per_kw = node["bos_per_kw"] ? node["bos_per_kw"].as<double>() : 0.;
        opex_per_kw = node["opex_per_kw"] ? node["opex_per_kw"].as<double>() : 0.;
        turbine_number = node["turbine_number"] ? node["turbine_number"].as<int>() : 0;
        labor_rate = node["labor_rate"] ? node["labor_rate"].as<double>() : 0.;
        painting_rate = node["painting_rate"] ? node["painting_rate"].as<double>() : 0.;
        blade_mass_cost_coeff = node["blade_mass_cost_coeff"] ? node["blade_mass_cost_coeff"].as<double>() : 0.;
        hub_mass_cost_coeff = node["hub_mass_cost_coeff"] ? node["hub_mass_cost_coeff"].as<double>() : 0.;
        pitch_system_mass_cost_coeff = node["pitch_system_mass_cost_coeff"] ? node["pitch_system_mass_cost_coeff"].as<double>() : 0.;
        spinner_mass_cost_coeff = node["spinner_mass_cost_coeff"] ? node["spinner_mass_cost_coeff"].as<double>() : 0.;
        lss_mass_cost_coeff = node["lss_mass_cost_coeff"] ? node["lss_mass_cost_coeff"].as<double>() : 0.;
        bearing_mass_cost_coeff = node["bearing_mass_cost_coeff"] ? node["bearing_mass_cost_coeff"].as<double>() : 0.;
        gearbox_mass_cost_coeff = node["gearbox_mass_cost_coeff"] ? node["gearbox_mass_cost_coeff"].as<double>() : 0.;
        hss_mass_cost_coeff = node["hss_mass_cost_coeff"] ? node["hss_mass_cost_coeff"].as<double>() : 0.;
        generator_mass_cost_coeff = node["generator_mass_cost_coeff"] ? node["generator_mass_cost_coeff"].as<double>() : 0.;
        bedplate_mass_cost_coeff = node["bedplate_mass_cost_coeff"] ? node["bedplate_mass_cost_coeff"].as<double>() : 0.;
        yaw_mass_cost_coeff = node["yaw_mass_cost_coeff"] ? node["yaw_mass_cost_coeff"].as<double>() : 0.;
        converter_mass_cost_coeff = node["converter_mass_cost_coeff"] ? node["converter_mass_cost_coeff"].as<double>() : 0.;
        transformer_mass_cost_coeff = node["transformer_mass_cost_coeff"] ? node["transformer_mass_cost_coeff"].as<double>() : 0.;
        hvac_mass_cost_coeff = node["hvac_mass_cost_coeff"] ? node["hvac_mass_cost_coeff"].as<double>() : 0.;
        cover_mass_cost_coeff = node["cover_mass_cost_coeff"] ? node["cover_mass_cost_coeff"].as<double>() : 0.;
        elec_connec_machine_rating_cost_coeff = node["elec_connec_machine_rating_cost_coeff"] ? node["elec_connec_machine_rating_cost_coeff"].as<double>() : 0.;
        platforms_mass_cost_coeff = node["platforms_mass_cost_coeff"] ? node["platforms_mass_cost_coeff"].as<double>() : 0.;
        tower_mass_cost_coeff = node["tower_mass_cost_coeff"] ? node["tower_mass_cost_coeff"].as<double>() : 0.;
        controls_machine_rating_cost_coeff = node["controls_machine_rating_cost_coeff"] ? node["controls_machine_rating_cost_coeff"].as<double>() : 0.;
        crane_cost = node["crane_cost"] ? node["crane_cost"].as<double>() : 0.;
        electricity_price = node["electricity_price"] ? node["electricity_price"].as<double>() : 0.;
        reserve_margin_price = node["reserve_margin_price"] ? node["reserve_margin_price"].as<double>() : 0.;
        capacity_credit = node["capacity_credit"] ? node["capacity_credit"].as<double>() : 0.;
        benchmark_price = node["benchmark_price"] ? node["benchmark_price"].as<double>() : 0.;
    }
};

// Configuration
struct Configuration {
    double wind_speed; // Average wind speed measured at hub height corresponding to the outputs
    double rotor_speed; // Rotor speed corresponding to the outputs
    double blade_pitch; // Collective blade pitch angle corresponding to the outputs
    double tip_speed_ratio; // Rotor tip speed ratio corresponding to the outputs

    void parse(const YAML::Node& node) {
        wind_speed = node["wind_speed"] ? node["wind_speed"].as<double>() : 0.;
        rotor_speed = node["rotor_speed"] ? node["rotor_speed"].as<double>() : 0.;
        blade_pitch = node["blade_pitch"] ? node["blade_pitch"].as<double>() : 0.;
        tip_speed_ratio = node["tip_speed_ratio"] ? node["tip_speed_ratio"].as<double>() : 0.;
    }
};

// Outputs integrated along blade span
struct Integrated {
    double mechanical_power; // Mechanical power of the rotor measured at the high speed shaft
    double electrical_power; // Eelectrical power of the rotor measured at the output of the generator
    double generator_torque; // Electrical torque of the generator
    double rotor_thrust; // Aerodynamic thrust of the rotor measured at the hub
    double rotor_torque; // Mechanical torque of the rotor measured at the hub

    void parse(const YAML::Node& node) {
        mechanical_power = node["mechanical_power"] ? node["mechanical_power"].as<double>() : 0.;
        electrical_power = node["electrical_power"] ? node["electrical_power"].as<double>() : 0.;
        generator_torque = node["generator_torque"] ? node["generator_torque"].as<double>() : 0.;
        rotor_thrust = node["rotor_thrust"] ? node["rotor_thrust"].as<double>() : 0.;
        rotor_torque = node["rotor_torque"] ? node["rotor_torque"].as<double>() : 0.;
    }
};

// Aerodynamic loading along the axial rotor direction
struct AeroForceAxial {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Aerodynamic loading along the tangential rotor direction
struct AeroForceTangential {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Blade deflection with respect to the undeflected configuration along the x axis for the pitching blade root coordinate system
struct BladeTranslationDeflectionXPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Blade deflection with respect to the undeflected configuration along the y axis for the pitching blade root coordinate system
struct BladeTranslationDeflectionYPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Blade deflection with respect to the undeflected configuration along the z axis for the pitching blade root coordinate system
struct BladeTranslationDeflectionZPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Blade rotation with respect to the undeflected configuration along the z axis for the pitching blade root coordinate system
struct BladeRotationDeflectionZPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Distribution along blade span of the reaction force along the x axis (edgewise shear). The force follows the pitching blade root coordinate system
struct BladeFxPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Distribution along blade span of the reaction force along the y axis (flapwise shear). The force follows the pitching blade root coordinate system
struct BladeFyPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Distribution along blade span of the reaction force along the z axis (axial). The force follows the pitching blade root coordinate system
struct BladeFzPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Distribution along blade span of the reaction moment along the x axis (flapwise moment). The moment follows the pitching blade root coordinate system
struct BladeMxPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Distribution along blade span of the reaction moment along the y axis (edgewise moment). The moment follows the pitching blade root coordinate system
struct BladeMyPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Distribution along blade span of the reaction moment along the z axis (torsional moment). The moment follows the pitching blade root coordinate system
struct BladeMzPitching {
    std::vector<double> grid;
    std::vector<double> values;

    void parse(const YAML::Node& node) {
        grid = node["grid"] ? node["grid"].as<std::vector<double>>() : std::vector<double>();
        values = node["values"] ? node["values"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Outputs distributed along blade span
struct Distributed {
    AeroForceAxial aero_force_axial; // Aerodynamic loading along the axial rotor direction
    AeroForceTangential aero_force_tangential; // Aerodynamic loading along the tangential rotor direction
    BladeTranslationDeflectionXPitching blade_translation_deflection_x_pitching; // Blade deflection with respect to the undeflected configuration along the x axis for the pitching blade root coordinate system
    BladeTranslationDeflectionYPitching blade_translation_deflection_y_pitching; // Blade deflection with respect to the undeflected configuration along the y axis for the pitching blade root coordinate system
    BladeTranslationDeflectionZPitching blade_translation_deflection_z_pitching; // Blade deflection with respect to the undeflected configuration along the z axis for the pitching blade root coordinate system
    BladeRotationDeflectionZPitching blade_rotation_deflection_z_pitching; // Blade rotation with respect to the undeflected configuration along the z axis for the pitching blade root coordinate system
    BladeFxPitching blade_fx_pitching; // Distribution along blade span of the reaction force along the x axis (edgewise shear). The force follows the pitching blade root coordinate system
    BladeFyPitching blade_fy_pitching; // Distribution along blade span of the reaction force along the y axis (flapwise shear). The force follows the pitching blade root coordinate system
    BladeFzPitching blade_fz_pitching; // Distribution along blade span of the reaction force along the z axis (axial). The force follows the pitching blade root coordinate system
    BladeMxPitching blade_mx_pitching; // Distribution along blade span of the reaction moment along the x axis (flapwise moment). The moment follows the pitching blade root coordinate system
    BladeMyPitching blade_my_pitching; // Distribution along blade span of the reaction moment along the y axis (edgewise moment). The moment follows the pitching blade root coordinate system
    BladeMzPitching blade_mz_pitching; // Distribution along blade span of the reaction moment along the z axis (torsional moment). The moment follows the pitching blade root coordinate system

    void parse(const YAML::Node& node) {
        if (node["aero_force_axial"]) {
            aero_force_axial.parse(node["aero_force_axial"]);
        }
        if (node["aero_force_tangential"]) {
            aero_force_tangential.parse(node["aero_force_tangential"]);
        }
        if (node["blade_translation_deflection_x_pitching"]) {
            blade_translation_deflection_x_pitching.parse(node["blade_translation_deflection_x_pitching"]);
        }
        if (node["blade_translation_deflection_y_pitching"]) {
            blade_translation_deflection_y_pitching.parse(node["blade_translation_deflection_y_pitching"]);
        }
        if (node["blade_translation_deflection_z_pitching"]) {
            blade_translation_deflection_z_pitching.parse(node["blade_translation_deflection_z_pitching"]);
        }
        if (node["blade_rotation_deflection_z_pitching"]) {
            blade_rotation_deflection_z_pitching.parse(node["blade_rotation_deflection_z_pitching"]);
        }
        if (node["blade_fx_pitching"]) {
            blade_fx_pitching.parse(node["blade_fx_pitching"]);
        }
        if (node["blade_fy_pitching"]) {
            blade_fy_pitching.parse(node["blade_fy_pitching"]);
        }
        if (node["blade_fz_pitching"]) {
            blade_fz_pitching.parse(node["blade_fz_pitching"]);
        }
        if (node["blade_mx_pitching"]) {
            blade_mx_pitching.parse(node["blade_mx_pitching"]);
        }
        if (node["blade_my_pitching"]) {
            blade_my_pitching.parse(node["blade_my_pitching"]);
        }
        if (node["blade_mz_pitching"]) {
            blade_mz_pitching.parse(node["blade_mz_pitching"]);
        }
    }
};

// Cases
struct Cases {
    Configuration configuration;
    Integrated integrated; // Outputs integrated along blade span
    Distributed distributed; // Outputs distributed along blade span
    std::vector<double> frequency_undamped; // Undamped natural frequencies of the system
    std::vector<double> damping_ratio; // Critical damping ratios of the modes characterizing the system

    void parse(const YAML::Node& node) {
        if (node["configuration"]) {
            configuration.parse(node["configuration"]);
        }
        if (node["integrated"]) {
            integrated.parse(node["integrated"]);
        }
        if (node["distributed"]) {
            distributed.parse(node["distributed"]);
        }
        frequency_undamped = node["frequency_undamped"] ? node["frequency_undamped"].as<std::vector<double>>() : std::vector<double>();
        damping_ratio = node["damping_ratio"] ? node["damping_ratio"].as<std::vector<double>>() : std::vector<double>();
    }
};

// Modal response and loads and performance metrics describing the turbine
struct Outputs {
    std::string solver; // Name and version of the numerical solver used to generate the outputs
    std::vector<Cases> cases; // Data points where outputs are computed at varying wind speed, rotor speed, etc.

    void parse(const YAML::Node& node) {
        solver = node["solver"] ? node["solver"].as<std::string>() : "";
        if (node["cases"]) {
            for (const auto& item : node["cases"]) {
                Cases x;
                x.parse(item);
                cases.push_back(x);
            }
        }
    }
};

// Turbine
struct Turbine {
    std::string comments; // Text field to describe the wind turbine, the changes to previous versions, etc,
    std::string name; // Unique identifier of the wind turbine model
    std::string windio_version; // Version of windIO used
    Assembly assembly; // The field assembly includes nine entries that aim at describing the overall configuration of the wind turbine
    Components components; // Nested dictionary structure of components describing the wind turbine assembly
    std::vector<Airfoils> airfoils; // Database of airfoils. windIO describes the airfoils in terms of coordinates, polars, and unsteady aerodynamic coefficients. The yaml entry airfoils consists of a list of elements. For each set of coordinates, multiple sets of polars and parameters can co-exist. Note that the airfoils listed in this database are not all necessarily used in the blade. Only the ones called in :code:`airfoil_position' within :code:`outer_shape_bem` of the :code:`component` :code:`blade` will actually be loaded to model the blade.
    std::vector<Materials> materials; // Database of the materials. The schema enforces that the fields :code:`name`, :code:`orth`, :code:`rho`, :code:`E`, and :code:`nu` are specified. For composites, direction 1 is aligned with the main fiber direction, direction 2 is in the plane transverse to the fibers, and direction 3 is perspendicular to the laminate plane. Note that fiber angles are specified in :code:`internal_structure_2d_fem` of the :code:`component` :code:`blade`.
    Control control; // Dictionary reporting the data describing the wind turbine controller
    Environment environment; // The field :code:`environment` includes the data characterizing air and water where the wind turbine operates.
    Bos bos; // Data for a balance of station cost analysis.
    Costs costs; // Data for a levelized cost of energy analysis.
    Outputs outputs; // Modal response and loads and performance metrics describing the turbine

    void parse(const YAML::Node& node) {
        comments = node["comments"] ? node["comments"].as<std::string>() : "";
        name = node["name"] ? node["name"].as<std::string>() : "";
        windio_version = node["windio_version"] ? node["windio_version"].as<std::string>() : "";
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
        if (node["outputs"]) {
            outputs.parse(node["outputs"]);
        }
    }
};


struct WindIO {
    Turbine turbine;
    WindIO(std::string file_path) {
        const YAML::Node config = YAML::Load(file_path);
        turbine.parse(config);
    }
};

}  // namespace openturbine::wind_io
