#pragma once

#include <array>
#include <cmath>

#include "types.hpp"

namespace openturbine {

/**
 * @brief Generates a 6x6 cross-sectional stiffness matrix for use in beam elements
 *
 * This function constructs the cross-sectional stiffness matrix that relates the generalized
 * strains to the generalized forces/moments in a beam cross-section. The matrix accounts for
 * coupling between axial, bending, shear, and torsional deformations due to offset centers
 * and principal axis orientations.
 *
 * @param EA Axial stiffness (E×A) [Force]
 * @param EI_x Bending stiffness around local x-axis in principal frame [Force × Length²]
 * @param EI_y Bending stiffness around local y-axis in principal frame [Force × Length²]
 * @param GKt Torsional stiffness (G×J) [Force × Length²]
 * @param GA Shear stiffness (G×A) [Force]
 * @param kxs Timoshenko shear correction factor in x-direction [dimensionless]
 * @param kys Timoshenko shear correction factor in y-direction [dimensionless]
 * @param x_C x-coordinate of elastic centroid relative to reference point [Length]
 * @param y_C y-coordinate of elastic centroid relative to reference point [Length]
 * @param theta_p Rotation angle from reference axes to principal bending axes [radians]
 * @param x_S x-coordinate of shear center relative to reference point [Length]
 * @param y_S y-coordinate of shear center relative to reference point [Length]
 * @param theta_s Rotation angle from reference axes to principal shear axes [radians]
 *
 * @return 6x6 cross-sectional stiffness matrix
 */
static Array_6x6 GenerateStiffnessMatrix(
    double EA, double EI_x, double EI_y, double GKt, double GA, double kxs, double kys, double x_C,
    double y_C, double theta_p, double x_S, double y_S, double theta_s
) {
    const double cos_theta_p = std::cos(theta_p);
    const double sin_theta_p = std::sin(theta_p);
    const double cos_theta_s = std::cos(theta_s);
    const double sin_theta_s = std::sin(theta_s);

    // Calculate bending stiffness in principal frame
    const double bending_xx = EI_x * cos_theta_p * cos_theta_p + EI_y * sin_theta_p * sin_theta_p;
    const double bending_yy = EI_x * sin_theta_p * sin_theta_p + EI_y * cos_theta_p * cos_theta_p;
    const double bending_xy = (EI_y - EI_x) * sin_theta_p * cos_theta_p;

    // Calculate shear stiffness in principal frame
    const double shear_xx = GA * (kxs * cos_theta_s * cos_theta_s + kys * sin_theta_s * sin_theta_s);
    const double shear_yy = GA * (kxs * sin_theta_s * sin_theta_s + kys * cos_theta_s * cos_theta_s);
    const double shear_xy = GA * (kys - kxs) * sin_theta_s * cos_theta_s;

    //--------------------------------------------------------------------------
    // Assemble stiffness matrix by blocks
    //--------------------------------------------------------------------------
    Array_6x6 stiffness_matrix = {};  // initialized to zeros

    // Shear-shear coupling (rows 0-1, cols 0-1)
    stiffness_matrix[0][0] = shear_xx;
    stiffness_matrix[0][1] = -shear_xy;
    stiffness_matrix[1][0] = -shear_xy;
    stiffness_matrix[1][1] = shear_yy;

    // Axial stiffness (row 2, col 2)
    stiffness_matrix[2][2] = EA;

    // Axial-bending coupling due to centroid offset (row 2, cols 3-4)
    stiffness_matrix[2][3] = EA * y_C;
    stiffness_matrix[2][4] = -EA * x_C;

    // Bending-axial coupling due to centroid offset (rows 3-4, col 2)
    stiffness_matrix[3][2] = EA * y_C;
    stiffness_matrix[4][2] = -EA * x_C;

    // Bending stiffness with axial-bending coupling (rows 3-4, cols 3-4)
    stiffness_matrix[3][3] = bending_xx + EA * y_C * y_C;
    stiffness_matrix[3][4] = -bending_xy - EA * x_C * y_C;
    stiffness_matrix[4][3] = -bending_xy - EA * x_C * y_C;
    stiffness_matrix[4][4] = bending_yy + EA * x_C * x_C;

    // Shear-torsion coupling due to shear center offset (rows 0-1, col 5)
    stiffness_matrix[0][5] = -shear_xx * y_S - shear_xy * x_S;
    stiffness_matrix[1][5] = shear_xy * y_S + shear_yy * x_S;

    // Torsion-shear coupling due to shear center offset (row 5, cols 0-1)
    stiffness_matrix[5][0] = -shear_xx * y_S - shear_xy * x_S;
    stiffness_matrix[5][1] = shear_xy * y_S + shear_yy * x_S;

    // Torsional stiffness with shear-torsion coupling (row 5, col 5)
    stiffness_matrix[5][5] =
        GKt + shear_xx * y_S * y_S + 2. * shear_xy * x_S * y_S + shear_yy * x_S * x_S;

    return stiffness_matrix;
}

}  // namespace openturbine
