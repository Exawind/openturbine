#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import argparse
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import yaml

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

#--------------------------------------------------------------------------------
# Helper functions
#--------------------------------------------------------------------------------

def extract_data(windio_data):
    """
    Extract all necessary data from windIO v2.0 file in a single pass.

    Returns:
        tuple: (airfoils, positions, polars, geometry_data)
    """
    airfoils = {}
    positions = []
    polars = {}
    geometry_data = {}
    ac_data = {}
    # Extract airfoil definitions
    if 'airfoils' in windio_data:
        for airfoil in windio_data['airfoils']:
            name = airfoil['name']
            coordinates = airfoil['coordinates']

            if isinstance(coordinates, dict) and 'x' in coordinates and 'y' in coordinates:
                airfoils[name] = {
                    'x': np.array(coordinates['x']),
                    'y': np.array(coordinates['y'])
                }
                print(f"Extracted {len(coordinates['x'])} coordinates for {name}")

            # Extract polar data
            if 'polars' in airfoil and airfoil['polars']:
                polar = airfoil['polars'][0]  # Take first polar configuration
                if 're_sets' in polar and polar['re_sets']:
                    re_set = polar['re_sets'][0]  # Take first Reynolds number set
                    if 'cl' in re_set and 'cd' in re_set:
                        polars[name] = {
                            'alpha': np.array(re_set['cl']['grid']),
                            'cl': np.array(re_set['cl']['values']),
                            'cd': np.array(re_set['cd']['values']),
                            're': re_set.get('re', None)
                        }

                        # Add moment coefficient if available
                        if 'cm' in re_set and 'values' in re_set['cm']:
                            polars[name]['cm'] = np.array(re_set['cm']['values'])

                        print(f"Extracted polar data for {name}")

            # Extract aerodynamic center data if available
            if 'aerodynamic_center_local_coordinates' in airfoil:
                ac_coords = airfoil['aerodynamic_center_local_coordinates']
                if isinstance(ac_coords, dict) and 'x' in ac_coords and 'y' in ac_coords:
                    ac_data[name] = {
                        'x': ac_coords['x'],
                        'y': ac_coords['y']
                    }
                    print(f"Extracted aerodynamic center for {name}: ({ac_coords['x']:.3f}, {ac_coords['y']:.3f})")

    # Extract blade positions and geometry
    if 'components' in windio_data:
        blade_data = windio_data['components']['blade']
        if 'outer_shape' in blade_data:
            outer_shape = blade_data['outer_shape']

            # Extract airfoil positions
            if 'airfoils' in outer_shape:
                for airfoil_entry in outer_shape['airfoils']:
                    positions.append({
                        'grid': airfoil_entry['spanwise_position'],
                        'airfoil_name': airfoil_entry['name']
                    })
                print(f"Found {len(positions)} airfoil positions")

            # Extract geometry data
            for prop in ['chord', 'twist', 'section_offset_y', 'section_offset_x', 'rthick']:
                if prop in outer_shape:
                    data = outer_shape[prop]
                    if 'grid' in data and 'values' in data:
                        geometry_data[prop] = {
                            'grid': np.array(data['grid']),
                            'values': np.array(data['values'])
                        }
                        print(f"Found {prop} data with {len(data['grid'])} points")

    return airfoils, positions, polars, geometry_data, ac_data

#-------------------------------------------------------------------------------
# Core visualization functions
#-------------------------------------------------------------------------------

def visualize_airfoils_3D(airfoils, positions, geometry_data, ac_data, output_file=None):
    """Create 3D visualization of airfoils along blade span with proper transformations.

    Args:
        airfoils (dict): Dictionary of airfoil data
        positions (list): List of dictionaries containing airfoil positions
        geometry_data (dict): Dictionary of geometry data
        ac_data (dict): Dictionary of aerodynamic center data
        output_file (str): Path to save the output file
    """
    print("Creating 3D airfoil visualization...")
    if not airfoils or not positions:
        print("No airfoil data or positions found.")
        return

    # Create matplotlib figure (static)
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create plotly figure (interactive)
    fig_interactive = go.Figure()

    for pos in positions:
        airfoil_name = pos['airfoil_name']
        span_position = pos['grid']

        if airfoil_name in airfoils:
            airfoil_data = airfoils[airfoil_name]
            x_coords, y_coords = airfoil_data['x'], airfoil_data['y']

            #-------------------------------------------------------------
            # Interpolate properties from geometry data
            #-------------------------------------------------------------
            chord = 1.
            twist = 0.
            section_offset_x = 0.
            section_offset_y = 0.
            thickness = 0.2

            if 'chord' in geometry_data:
                chord = np.interp(span_position, geometry_data['chord']['grid'],
                                geometry_data['chord']['values'])
            if 'twist' in geometry_data:
                twist = np.interp(
                    span_position, geometry_data['twist']['grid'], geometry_data['twist']['values']
                )
            if 'section_offset_x' in geometry_data:
                section_offset_x = np.interp(
                    span_position, geometry_data['section_offset_x']['grid'],
                    geometry_data['section_offset_x']['values']
                )
            if 'section_offset_y' in geometry_data:
                section_offset_y = np.interp(
                    span_position, geometry_data['section_offset_y']['grid'],
                    geometry_data['section_offset_y']['values']
                )
            if 'rthick' in geometry_data:
                thickness = np.interp(
                    span_position, geometry_data['rthick']['grid'], geometry_data['rthick']['values']
                )

            """
            print(f"Span position: {span_position}")
            closest_idx = np.abs(geometry_data['chord']['grid'] - span_position).argmin()
            closest_chord = geometry_data['chord']['values'][closest_idx]
            print(f"Closest chord value: {closest_chord}, Interpolated chord: {chord}")
            closest_twist = geometry_data['twist']['values'][closest_idx]
            print(f"Closest twist value: {closest_twist}, Interpolated twist: {twist}")
            closest_section_offset_y = geometry_data['section_offset_y']['values'][closest_idx]
            print(f"Closest section_offset_y value: {closest_section_offset_y}, Interpolated section_offset_y: {section_offset_y}")
            closest_thickness = geometry_data['rthick']['values'][closest_idx]
            print(f"Closest thickness value: {closest_thickness}, Interpolated thickness: {thickness}")
            """

            #-------------------------------------------------------------
            # Apply transformations to the airfoils
            #-------------------------------------------------------------

            # Step 1: Scale airfoil by chord length
            x_scaled = x_coords * chord
            y_scaled = y_coords * chord

            # Step 2: Find leading edge
            leading_edge_idx = np.argmin(x_scaled)
            x_le = x_scaled[leading_edge_idx]
            y_le = y_scaled[leading_edge_idx]

            # Step 3: Apply section offsets to all points
            # offset_y is along chord direction (x-axis), offset_x is normal to chord (y-axis)
            x_with_offsets = x_scaled + section_offset_y
            y_with_offsets = y_scaled + section_offset_x

            # Step 4: Apply twist rotation around leading edge
            twist_rad = np.radians(twist)
            # Rotation matrix for 2D rotation
            R = np.array([
                [np.cos(twist_rad), -np.sin(twist_rad)],
                [np.sin(twist_rad), np.cos(twist_rad)]
            ])

            # Apply rotation to all airfoil points
            x_final = np.zeros_like(x_with_offsets)
            y_final = np.zeros_like(y_with_offsets)

            for i in range(len(x_with_offsets)):
                # Vector from leading edge to point
                p = np.array([x_with_offsets[i] - x_le, y_with_offsets[i] - y_le])
                # Rotate the vector
                p_rotated = R @ p
                # Final coordinates: leading edge + rotated vector
                x_final[i] = x_le + p_rotated[0]
                y_final[i] = y_le + p_rotated[1]

            # Step 5: Position along blade span
            z_coords = np.full_like(x_final, span_position)

            #-------------------------------------------------------------
            # Add chord line and aerodynamic center
            #-------------------------------------------------------------
            # Apply the same transformations to chord line as we did to the airfoil

            # Find leading edge again for chord line
            x_le_original, y_le_original = x_scaled[leading_edge_idx], y_scaled[leading_edge_idx]
            chord_line_x = np.array([x_le_original, x_le_original + chord])
            chord_line_y = np.array([y_le_original, y_le_original])
            chord_line_z = np.full_like(chord_line_x, span_position)

            # Apply section offsets to chord line
            chord_line_x_with_offsets = chord_line_x + section_offset_y
            chord_line_y_with_offsets = chord_line_y + section_offset_x

            # Apply twist rotation to chord line
            chord_line_x_final = np.zeros_like(chord_line_x_with_offsets)
            chord_line_y_final = np.zeros_like(chord_line_y_with_offsets)

            for i in range(len(chord_line_x_with_offsets)):
                p = np.array([chord_line_x_with_offsets[i] - x_le_original, chord_line_y_with_offsets[i] - y_le_original])
                p_rotated = R @ p
                chord_line_x_final[i] = x_le_original + p_rotated[0]
                chord_line_y_final[i] = y_le_original + p_rotated[1]

            # Add aerodynamic center dot if available (already transformed)
            if airfoil_name in ac_data:
                ac_x = ac_data[airfoil_name]['x']
                ac_y = ac_data[airfoil_name]['y']
                ac_z = span_position

            #-------------------------------------------------------------
            # Matplotlib plot
            #-------------------------------------------------------------
            line_width = 2
            ax.plot(
                z_coords, x_final, y_final,
                label=f'{airfoil_name} (r={span_position:.1f}m, c={chord:.2f}m, t/c={thickness:.3f})',
                linewidth=line_width
            )
            # Add chord line
            ax.plot(
                chord_line_z, chord_line_x_final, chord_line_y_final,
                color='gray', linestyle='--', linewidth=1
            )
            # Add aerodynamic center point if available
            if airfoil_name in ac_data:
                ax.scatter(
                    [ac_z], [ac_x], [ac_y],
                    color='red', marker='o', s=5
                )

            #-------------------------------------------------------------
            # Plotly plot
            #-------------------------------------------------------------
            fig_interactive.add_trace(go.Scatter3d(
                x=z_coords, y=x_final, z=y_final,
                mode='lines',
                name=f'{airfoil_name} (r={span_position:.1f}m, c={chord:.2f}m, t/c={thickness:.3f})',
                line=dict(width=line_width),
                hovertemplate=f'<b>{airfoil_name}</b><br>' +
                            f'r/R: {span_position:.1f}m<br>' +
                            f'Chord: {chord:.2f}m<br>' +
                            f't/c: {thickness:.3f}<br>' +
                            f'Twist: {twist:.1f}°<br>' +
                            f'Offset: ({section_offset_x:.3f}, {section_offset_y:.3f})<extra></extra>'
            ))
            # Plotly plot - chord line
            fig_interactive.add_trace(go.Scatter3d(
                x=chord_line_z, y=chord_line_x_final, z=chord_line_y_final,
                mode='lines',
                name=f'Chord line {airfoil_name}',
                line=dict(width=1, dash='dash', color='gray'),
                showlegend=False,
                hovertemplate=f'<b>Chord line {airfoil_name}</b><extra></extra>'
            ))
            # Plotly plot - aerodynamic center
            if airfoil_name in ac_data:
                fig_interactive.add_trace(go.Scatter3d(
                    x=[ac_z], y=[ac_x], z=[ac_y],
                    mode='markers',
                    name=f'AC {airfoil_name}',
                    marker=dict(size=2, color='red', symbol='circle'),
                    showlegend=False,
                    hovertemplate=f'<b>Aerodynamic Center {airfoil_name}</b><br>' +
                                f'Local coords: ({ac_x:.3f}, {ac_y:.3f})<extra></extra>'
                ))

    #-------------------------------------------------------------
    # Customize and save plots
    #-------------------------------------------------------------
    ax.set_xlabel('Spanwise Position, r/R (-)')
    ax.set_ylabel('Chord (m)')
    ax.set_zlabel('Thickness (m)')
    ax.set_title('3D Airfoil Cross-sections Along Blade Span (with transformations)')
    ax.set_box_aspect([20, 1, 1])
    ax.view_init(elev=20, azim=45)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    fig_interactive.update_layout(
        title='3D Airfoil Cross-sections Along Blade Span (Interactive)',
        scene=dict(
            xaxis_title='Spanwise Position, r/R (-)',
            yaxis_title='Chord (m)',
            zaxis_title='Thickness (m)',
            aspectmode='manual',
            aspectratio=dict(x=20, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=2000, height=1000,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
    )

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Static figure saved to: {output_file}")

        base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
        extension = output_file.rsplit('.', 1)[1] if '.' in output_file else 'html'
        interactive_output = f"{base_name}_interactive.{extension}"
    else:
        plt.savefig("airfoil_3d_visualization.png", dpi=300, bbox_inches='tight')
        interactive_output = "airfoil_3d_visualization_interactive.html"

    fig_interactive.write_html(interactive_output)
    print(f"Interactive figure saved to: {interactive_output}")
    plt.close(fig)
    print("3D visualization complete!")

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------

def main():
    """Main function to handle command line arguments and run the visualization.

    Example usage (run from the build directory):
        python ../src/utilities/scripts/visualize_windio_airfoils.py \
            tests/regression_tests/interfaces/interfaces_test_files/IEA-15-240-RWT_v2.yaml
    """
    parser = argparse.ArgumentParser(
        description='Visualize airfoils from a WindIO v2.0 YAML file'
    )

    #-------------------------------------------------------------
    # Input/output arguments
    #-------------------------------------------------------------

    # WindIO input file -- required argument
    parser.add_argument(
        'yaml_file',
        help='Path to the WindIO v2.0 YAML file'
    )

    # Output file -- optional argument
    parser.add_argument(
        '--output', '-o',
        help='Output file path for the figure (optional)'
    )

    # Polar plots -- optional argument
    parser.add_argument(
        '--polars', '-p',
        choices=['2d', '3d', '3d_interactive', 'surface'],
        help='Generate polar plots (2d, 3d, 3d_interactive, or surface)'
    )

    # Polar output file -- optional argument
    parser.add_argument(
        '--polars-output',
        help='Output file path for the polar figure (optional)'
    )

    args = parser.parse_args()

    #-------------------------------------------------------------
    # Visualize windIO file
    #-------------------------------------------------------------

    if not Path(args.yaml_file).exists():
        print(f"Error: File '{args.yaml_file}' not found.")
        return

    try:
        with open(args.yaml_file, 'r') as file:
            windio_data = yaml.safe_load(file)
        print(f"Successfully loaded WindIO file")
        airfoils, positions, polars, geometry_data, ac_data = extract_data(windio_data)

        # Create 3D visualization of airfoil profiles
        visualize_airfoils_3D(airfoils, positions, geometry_data, ac_data, args.output)

        # Create polar visualizations
        if args.polars:
            visualize_polars(polars, positions, args.polars_output, args.polars)

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    exit(main())
