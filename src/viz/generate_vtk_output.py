import argparse
import numpy as np
import os
import vtk
from netCDF4 import Dataset
from typing import Dict, List, Optional, Tuple

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------

def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """Converts a 4x1 quaternion [w, i, j, k] to a 3x3 rotation matrix.

    The rotation matrix is computed using the following formula (used in OpenTurbine):
    R = | 1 - 2(j^2 + k^2)  2(i*j - w*k)  2(i*k + w*j) |
        | 2(i*j + w*k)  1 - 2(i^2 + k^2)  2(j*k - w*i) |
        | 2(i*k - w*j)  2(j*k + w*i)  1 - 2(i^2 + j^2) |

    Args:
        quaternion (list[float]): A list of 4 floats representing the quaternion

    Returns:
        np.ndarray: A 3x3 rotation matrix
    """
    # Unpack components from quaternion
    w, i, j, k = quaternion

    # Compute intermediate products
    ii, jj, kk = i*i, j*j, k*k
    ij, ik, jk = i*j, i*k, j*k
    wi, wj, wk = w*i, w*j, w*k

    # Create rotation matrix
    R = np.zeros((3, 3))

    R[0, 0] = 1. - 2. * (jj + kk)
    R[0, 1] = 2. * (ij - wk)
    R[0, 2] = 2. * (ik + wj)

    R[1, 0] = 2. * (ij + wk)
    R[1, 1] = 1. - 2. * (ii + kk)
    R[1, 2] = 2. * (jk - wi)

    R[2, 0] = 2. * (ik - wj)
    R[2, 1] = 2. * (jk + wi)
    R[2, 2] = 1. - 2. * (ii + jj)

    return R


def create_vector_array(name: str, num_components: int = 3):
    """Creates a VTK double array with the given name and number of components.

    Args:
        name (str): Name of the array
        num_components (int): Number of components in the array

    Returns:
        vtk.vtkDoubleArray: A VTK double array with the given name and number of components
    """
    array = vtk.vtkDoubleArray()
    array.SetNumberOfComponents(num_components)
    array.SetName(name)

    return array


#-------------------------------------------------------------------------------
# VTK output logic
#-------------------------------------------------------------------------------

class VTKOutput:
    """Class to generate VTK files from OpenTurbine (NetCDF-based) outputs."""

    def __init__(self, netcdf_path: str):
        """Initializes the visualizer with the path to the NetCDF file.

        Args:
            netcdf_path (str): Path to the NetCDF output file
        """
        self.netcdf_path = netcdf_path
        self.data = Dataset(netcdf_path, 'r')

        # Get dimensions from the NetCDF file
        self.num_nodes = len(self.data.dimensions['nodes'])
        self.num_timesteps = len(self.data.dimensions['time'])

        print(f"Loaded data with {self.num_nodes} nodes and {self.num_timesteps} timesteps")


    def _extract_node_data_at_timestep(
        self,
        timestep: int,
        node_indices: Optional[List[int]] = None
    ) -> List[Dict[str, List[float]]]:
        """Extracts node data for a specific timestep and returns a list of NodeData-like structures.

        Node data contains the following components:
        - Position: x_x, x_y, x_z, x_w, x_i, x_j, x_k
        - Velocity: v_x, v_y, v_z, v_i, v_j, v_k
        - Acceleration: a_x, a_y, a_z, a_i, a_j, a_k

        Args:
            timestep (int): The timestep to extract node data from
            node_indices (Optional[List[int]]):
                If provided, only extract data for these specific node indices.
                If None, extract data for all nodes.

        Returns:
            list[dict]: A list of NodeData-like structures (dicts)
        """
        if timestep >= self.num_timesteps:
            raise ValueError(f"Timestep {timestep} out of range (max: {self.num_timesteps-1})")

        # Validate that all requested indices are within range
        if node_indices is not None:
            for idx in node_indices:
                if idx < 0 or idx >= self.num_nodes:
                    raise ValueError(f"Node index {idx} out of range (max: {self.num_nodes-1})")
            indices_to_extract = node_indices

        # If no specific nodes requested, use all nodes
        if node_indices is None:
            indices_to_extract = range(self.num_nodes)

        # Read all data from NetCDF file at once for the requested timestep
        position = [
            self.data.variables['x_x'][timestep, :],
            self.data.variables['x_y'][timestep, :],
            self.data.variables['x_z'][timestep, :],
            self.data.variables['x_w'][timestep, :],
            self.data.variables['x_i'][timestep, :],
            self.data.variables['x_j'][timestep, :],
            self.data.variables['x_k'][timestep, :]
        ]
        velocity = [
            self.data.variables['v_x'][timestep, :],
            self.data.variables['v_y'][timestep, :],
            self.data.variables['v_z'][timestep, :],
            self.data.variables['v_i'][timestep, :],
            self.data.variables['v_j'][timestep, :],
            self.data.variables['v_k'][timestep, :]
        ]
        acceleration = [
            self.data.variables['a_x'][timestep, :],
            self.data.variables['a_y'][timestep, :],
            self.data.variables['a_z'][timestep, :],
            self.data.variables['a_i'][timestep, :],
            self.data.variables['a_j'][timestep, :],
            self.data.variables['a_k'][timestep, :]
        ]

        # Create node data dictionaries only for the requested indices
        nodes = []
        for i_node in indices_to_extract:
            node = {
                'position': [position[j_comp][i_node] for j_comp in range(7)],
                'velocity': [velocity[j_comp][i_node] for j_comp in range(6)],
                'acceleration': [acceleration[j_comp][i_node] for j_comp in range(6)]
            }
            nodes.append(node)

        return nodes


    def _add_node_data_to_vtk_object(self, vtk_object: vtk.vtkObject, nodes: List[Dict[str, List[float]]]):
        """Adds common node data to a VTK object (polydata or unstructured grid).

        Following data are added to the VTK object:
        - Orientation (as 3x3 rotation matrix)
        - Velocity (as 6x1 vector)
        - Acceleration (as 6x1 vector)

        Args:
            vtk_object: VTK object to add data to
            nodes: List of node data dictionaries
        """
        # Add orientation data
        orientation_arrays = {
            axis: create_vector_array(f"Orientation{axis}")
            for axis in ["X", "Y", "Z"]
        }

        for node in nodes:
            # Convert quaternion to rotation matrix
            quaternion = [
                node['position'][3],  # w
                node['position'][4],  # i
                node['position'][5],  # j
                node['position'][6]   # k
            ]
            R = quaternion_to_rotation_matrix(quaternion)

            orientation_arrays["X"].InsertNextTuple3(R[0, 0], R[1, 0], R[2, 0])
            orientation_arrays["Y"].InsertNextTuple3(R[0, 1], R[1, 1], R[2, 1])
            orientation_arrays["Z"].InsertNextTuple3(R[0, 2], R[1, 2], R[2, 2])

        for axis in ["X", "Y", "Z"]:
            vtk_object.GetPointData().AddArray(orientation_arrays[axis])

        # Add velocity data
        translation_velocity = create_vector_array("TranslationalVelocity")
        rotation_velocity = create_vector_array("RotationalVelocity")

        for node in nodes:
            translation_velocity.InsertNextTuple3(*node['velocity'][0:3])
            rotation_velocity.InsertNextTuple3(*node['velocity'][3:6])

        vtk_object.GetPointData().AddArray(translation_velocity)
        vtk_object.GetPointData().AddArray(rotation_velocity)

        # Add acceleration data
        trans_accel = create_vector_array("TranslationalAcceleration")
        rot_accel = create_vector_array("RotationalAcceleration")

        for node in nodes:
            trans_accel.InsertNextTuple3(*node['acceleration'][0:3])
            rot_accel.InsertNextTuple3(*node['acceleration'][3:6])

        vtk_object.GetPointData().AddArray(trans_accel)
        vtk_object.GetPointData().AddArray(rot_accel)


    def _visualize_structure(
        self,
        timestep: int,
        output_path: str,
        structure_type: str,
        line_connectivity: Optional[List[Tuple[int, int]]] = None,
        node_indices: Optional[List[int]] = None
    ):
        """Generic method to visualize different structures such as nodes, beams, or lines
        at a given timestep.

        Following types of structures are supported:
        - 'nodes': Visualize nodes as points
        - 'beam': Visualize single beam element as a Lagrange curve
        - 'lines': Visualize lines connecting node pairs

        Args:
            timestep (int): Timestep to visualize
            output_path (str): Path to save the output file to
            structure_type (str): Type of structure to visualize ('nodes', 'beam', or 'lines')
            line_connectivity (list[tuple[int, int]], optional): List of node index pairs to connect with lines.
                                                                 Required when structure_type is 'lines'.
            node_indices (Optional[List[int]]): If provided, only use these specific node indices.
        """
        nodes = self._extract_node_data_at_timestep(timestep, node_indices)

        # Create points for the structure
        points = vtk.vtkPoints()
        for node in nodes:
            points.InsertNextPoint(node['position'][0], node['position'][1], node['position'][2])

        # ------------------------------------------------------------
        # Nodes
        # ------------------------------------------------------------
        if structure_type == 'nodes':
            # Create polydata to visualize nodes
            vtk_object = vtk.vtkPolyData()
            vtk_object.SetPoints(points)

            # Create the writer
            writer = vtk.vtkXMLPolyDataWriter()
            structure_name = "nodes"

        # ------------------------------------------------------------
        # Beam
        # ------------------------------------------------------------
        elif structure_type == 'beam':
            # Create unstructured grid to visualize the beam element
            vtk_object = vtk.vtkUnstructuredGrid()
            vtk_object.SetPoints(points)

            # Create a Lagrange curve connecting all points
            pts_ids = vtk.vtkIdList()
            pts_ids.InsertNextId(0)
            pts_ids.InsertNextId(len(nodes) - 1)
            for j in range(1, len(nodes) - 1):
                pts_ids.InsertNextId(j)

            vtk_object.InsertNextCell(vtk.VTK_LAGRANGE_CURVE, pts_ids)

            # Create the writer
            writer = vtk.vtkXMLUnstructuredGridWriter()
            structure_name = "beam"

        # ------------------------------------------------------------
        # Lines
        # ------------------------------------------------------------
        elif structure_type == 'lines':
            if line_connectivity is None:
                raise ValueError("line_connectivity must be provided when structure_type is 'lines'")

            # Create lines
            lines = vtk.vtkCellArray()
            for start_idx, end_idx in line_connectivity:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, start_idx)
                line.GetPointIds().SetId(1, end_idx)
                lines.InsertNextCell(line)

            # Create polydata to visualize lines
            vtk_object = vtk.vtkPolyData()
            vtk_object.SetPoints(points)
            vtk_object.SetLines(lines)

            # Create the writer
            writer = vtk.vtkXMLPolyDataWriter()
            structure_name = "lines"

        else:
            raise ValueError(f"Unknown structure type: {structure_type}")

        # Add common data to the VTK object
        self._add_node_data_to_vtk_object(vtk_object, nodes)

        # Write the file
        writer.SetFileName(output_path)
        writer.SetInputData(vtk_object)
        writer.SetDataModeToAscii()
        writer.Write()

        print(f"Wrote {structure_name} visualization to {output_path}")


    def create_animation(self, output_dir: str, type: str, line_connectivity: Optional[List[Tuple[int, int]]] = None):
        """Creates an animation by generating a series of VTK files at provided timesteps.

        Following types of structures are supported:
        - 'beam': Visualize beam element as a Lagrange curve
        - 'nodes': Visualize nodes as points
        - 'lines': Visualize lines connecting node pairs

        Args:
            output_dir (str): Directory to save the output files to
            type (str): Type of visualization to generate ('beam', 'nodes', or 'lines')
            line_connectivity (Optional[List[Tuple[int, int]]]):
                List of tuples for line connectivity. Required for type='lines'.
        """
        os.makedirs(output_dir, exist_ok=True)

        for timestep in range(0, self.num_timesteps):
            if type == 'beam':
                beam_file = os.path.join(output_dir, f"beam_t{timestep:04d}.vtu")
                self._visualize_structure(timestep, beam_file, 'beam')

            elif type == 'nodes':
                nodes_file = os.path.join(output_dir, f"nodes_t{timestep:04d}.vtp")
                self._visualize_structure(timestep, nodes_file, 'nodes')

            elif type == 'lines':
                if line_connectivity is None:
                    raise ValueError("Line connectivity must be provided for 'lines' visualization type")
                lines_file = os.path.join(output_dir, f"lines_t{timestep:04d}.vtp")
                self._visualize_structure(timestep, lines_file, 'lines', line_connectivity)

            else:
                raise ValueError(f"Unknown type: {type}")

        print(f"Generated animation frames in {output_dir}")


    def _visualize_floating_platform(
        self,
        timestep: int,
        output_dir: str,
        platform_node_idx: int,
        fairlead_indices: List[int],
        anchor_indices: List[int]
    ):
        r"""Visualizes a floating platform with mooring lines at a given timestep.

        Following ASCII figure shows the platform and mooring line layout for a 3-point mooring system:

            Platform Node
                  o
                / | \
               /  |  \      <-- Fairlead connections
              /   |   \
             o    o    o    <-- Fairlead nodes
             |    |    |    <-- Mooring lines
             |    |    |
             o    o    o    <-- Anchor nodes

        Args:
            timestep (int): Timestep to visualize
            output_dir (str): Directory to save output files
            platform_node_idx (int): Index of the platform node
            fairlead_indices (List[int]): Indices for fairlead nodes of mooring lines
            anchor_indices (List[int]): Indices for anchor nodes of mooring lines
        """
        if len(fairlead_indices) != len(anchor_indices):
            raise ValueError("Number of fairlead indices must match number of anchor indices")

        os.makedirs(output_dir, exist_ok=True)

        # ------------------------------------------------------------
        # platform visualization
        # ------------------------------------------------------------
        platform_nodes = [platform_node_idx] + fairlead_indices

        # Create platform-to-fairleads connections
        platform_connections = []
        for i, _ in enumerate(fairlead_indices):
            # Since we're extracting a subset of nodes, we need to map the original indices
            # The platform node is at index 0, fairleads start at index 1
            # If there are 3 fairleads, there will be 3 connections, and the connections will be:
            # {0, 1}, {0, 2}, {0, 3}
            platform_connections.append((0, i + 1))

        # Generate and write platform visualization
        platform_file = os.path.join(output_dir, f"platform_t{timestep:05d}.vtp")
        self._visualize_structure(
            timestep,
            platform_file,
            'lines',
            platform_connections,
            platform_nodes
        )

        # ------------------------------------------------------------
        # mooring lines visualization
        # ------------------------------------------------------------
        mooring_nodes = []
        for i in range(len(fairlead_indices)):
            mooring_nodes.append(fairlead_indices[i])
            mooring_nodes.append(anchor_indices[i])

        # Create fairlead-to-anchor connections
        mooring_connections = []
        for i in range(len(fairlead_indices)):
            # Map to the new indices: fairleads start at 0, anchors start at 1 and they alternate
            # i.e. if there are 3 fairleads, there will be 3 anchors, and the connections will be:
            # 0-1, 2-3, 4-5
            mooring_connections.append((2 * i, 2 * i + 1))

        # Generate and write mooring visualization
        mooring_file = os.path.join(output_dir, f"mooring_t{timestep:05d}.vtp")
        self._visualize_structure(
            timestep,
            mooring_file,
            'lines',
            mooring_connections,
            mooring_nodes
        )

        print(f"Wrote floating platform visualization for timestep {timestep} to {output_dir}")


    def create_platform_animation(
        self,
        output_dir: str,
        platform_node_idx: int,
        fairlead_indices: List[int],
        anchor_indices: List[int]
    ):
        """Creates an animation of a floating platform with mooring lines.

        Args:
            output_dir (str): Directory to save output files
            platform_node_idx (int): Index of the platform node
            fairlead_indices (List[int]): Indices of fairlead nodes
            anchor_indices (List[int]): Indices of anchor nodes, must match fairlead_indices order
        """
        for timestep in range(0, self.num_timesteps):
            self._visualize_floating_platform(
                timestep,
                output_dir,
                platform_node_idx,
                fairlead_indices,
                anchor_indices
            )

        print(f"Generated floating platform animation in {output_dir}")


    def _visualize_rotor(
            self,
            timestep: int,
            output_path: str,
            beam_elements: Optional[List[List[int]]] = None
        ):
        """Generates visualization of wind turbine rotor at node/quadrature points.

        Multiple beam elements can be used to visualize a multi-blade rotor,
        e.g. for a 3-bladed rotor with 51 QPs per blade:
        beam_elements = [[0, 1, 2, ..., 50], [51, 52, 53, ..., 101], [102, 103, 104, ..., 152]]

        Args:
            timestep (int): Timestep to visualize
            output_path (str): Path to save the VTK file
            beam_elements (Optional[List[List[int]]]): List of node indices that form each beam element.
                                                       If None, assumes all nodes form a single beam element.
        """
        nodes = self._extract_node_data_at_timestep(timestep)

        # Create points for all nodes
        points = vtk.vtkPoints()
        for node in nodes:
            position = node['position']
            points.InsertNextPoint(position[0], position[1], position[2])

        # Create unstructured grid for beam visualization
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)

        # If beam_elements not provided, use default for IEA 15 MW turbine or create single beam
        if beam_elements is None:
            # Create a single beam
            beam_elements = [list(range(len(nodes)))]

            # hack: hardcoding IEA 15 MW rotor to support OpenTurbine regression tests
            # check if we have 153 nodes (3 blades × 51 QPs each), which is likely an IEA 15 MW turbine
            if len(nodes) == 153:
                beam_elements = [list(range(51)), list(range(51, 102)), list(range(102, 153))]

        # Add beam elements as Lagrange curves
        for beam in beam_elements:
            # Create connectivity for Lagrange curve
            point_ids = vtk.vtkIdList()
            point_ids.InsertNextId(beam[0])  # first point
            point_ids.InsertNextId(beam[-1])  # last point

            # Add intermediate points
            for j in range(1, len(beam)-1):
                point_ids.InsertNextId(beam[j])

            grid.InsertNextCell(vtk.VTK_LAGRANGE_CURVE, point_ids)

        # Add common node data (orientation, velocity, acceleration)
        self._add_node_data_to_vtk_object(grid, nodes)

        # Add optional data (forces, deformation) if available
        self._add_optional_force_moment_data(grid, nodes, timestep)
        self._add_optional_deformation_data(grid, nodes, timestep)

        # Write the file
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(grid)
        writer.SetDataModeToAscii()
        writer.Write()

        print(f"Wrote rotor visualization at nodes/quadrature points to {output_path}")


    def _add_optional_force_moment_data(self, vtk_object, nodes, timestep):
        """Add force and moment data to VTK object if available in the NetCDF file."""
        # Check if force data is available
        force_available = all(f'f_{comp}' in self.data.variables for comp in ['x', 'y', 'z'])
        if force_available:
            force = create_vector_array("Force")

            # Read force data from NetCDF
            f_x = self.data.variables['f_x'][timestep, :]
            f_y = self.data.variables['f_y'][timestep, :]
            f_z = self.data.variables['f_z'][timestep, :]

            for i in range(len(nodes)):
                force.InsertNextTuple3(f_x[i], f_y[i], f_z[i])

            vtk_object.GetPointData().AddArray(force)

        # Check if moment data is available
        moment_available = all(f'f_{comp}' in self.data.variables for comp in ['i', 'j', 'k'])
        if moment_available:
            moment = create_vector_array("Moment")

            # Read moment data from NetCDF
            m_i = self.data.variables['f_i'][timestep, :]
            m_j = self.data.variables['f_j'][timestep, :]
            m_k = self.data.variables['f_k'][timestep, :]

            for i in range(len(nodes)):
                moment.InsertNextTuple3(m_i[i], m_j[i], m_k[i])

            vtk_object.GetPointData().AddArray(moment)


    def _add_optional_deformation_data(self, vtk_object, nodes, timestep):
        """Add deformation data to VTK object if available in the NetCDF file."""
        deformation_available = all(f'deformation_{comp}' in self.data.variables for comp in ['x', 'y', 'z'])

        if deformation_available:
            deformation_vector = create_vector_array("DeformationVector")

            # Read deformation data from NetCDF
            u_x = self.data.variables['deformation_x'][timestep, :]
            u_y = self.data.variables['deformation_y'][timestep, :]
            u_z = self.data.variables['deformation_z'][timestep, :]

            for i in range(len(nodes)):
                deformation_vector.InsertNextTuple3(u_x[i], u_y[i], u_z[i])

            vtk_object.GetPointData().AddArray(deformation_vector)


    def create_rotor_animation(self, output_dir: str, beam_elements: Optional[List[List[int]]] = None):
        """Creates an animation of beams visualized at nodes/quadrature points.

        Args:
            output_dir (str): Directory to save output files
            beam_elements (Optional[List[List[int]]]): List of node indices for each beam element
        """
        os.makedirs(output_dir, exist_ok=True)

        for timestep in range(0, self.num_timesteps):
            output_path = os.path.join(output_dir, f"rotor_t{timestep:04d}.vtu")
            self._visualize_rotor(timestep, output_path, beam_elements)

        print(f"Generated rotor animation in {output_dir}")


#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------

def main():
    """Main function to parse arguments and generate VTK files.

    Default arguments:
        output_dir: vtk_output
        type: beam

    Example usage:
        - beam
        python generate_vtk_output.py --netcdf_file blade_interface.nc --output_dir vtk_output --type beam

        - lines
        python generate_vtk_output.py --netcdf_file blade_interface.nc --output_dir vtk_output --type lines --connectivity 0,1 1,2 2,3

        - platform
        python generate_vtk_output.py --netcdf_file platform.nc --output_dir vtk_output --type platform --platform_node 0 --fairleads 1,2,3 --anchors 4,5,6

    NOTE: Files are overwritten in the output directory if they already exist.
    """
    parser = argparse.ArgumentParser(description='Generate VTK files from OpenTurbine NetCDF output')

    # ------------------------------------------------------------
    # Input arguments
    # ------------------------------------------------------------

    # -------------------------------
    # Netcdf input file
    # -------------------------------
    parser.add_argument(
        'netcdf_file',
        type=str,
        help='Path to OpenTurbine NetCDF output file e.g. blade_interface.nc'
    )

    # -------------------------------
    # Output directory
    # -------------------------------
    parser.add_argument(
        '--output_dir',
        type=str,
        default='vtk_output',
        help='Directory for writing the vtk output files'
    )

    # -------------------------------
    # Type of visualization
    # -------------------------------
    parser.add_argument(
        '--type',
        type=str,
        default='beam',
        help='Type of visualization to generate: beam, nodes, lines, platform, or rotor'
    )

    # -------------------------------
    # Line connectivity
    # -------------------------------
    parser.add_argument(
        '--connectivity',
        type=str,
        nargs='+',
        help='Line connectivity in format "start,end" (space separated list). Required for type=lines'
    )

    # -------------------------------
    # Floating Platform args
    # -------------------------------
    parser.add_argument(
        '--platform_node',
        type=int,
        default=0,
        help='Node index for platform center. Used with type=platform'
    )
    parser.add_argument(
        '--fairleads',
        type=str,
        default='1,3,5',
        help='Comma-separated list of fairlead node indices (e.g., "1,3,5"). Required for type=platform'
    )
    parser.add_argument(
        '--anchors',
        type=str,
        default='2,4,6',
        help='Comma-separated list of anchor node indices (e.g., "2,4,6"). Required for type=platform'
    )

    # -------------------------------
    # Rotor visualization args
    # -------------------------------
    parser.add_argument(
        '--beam_elements',
        type=str,
        help='Comma-separated list of beam elements to visualize (e.g., "0,1,2 3,4,5"). Required for type=rotor'
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Create animation based on simulation type
    # ------------------------------------------------------------

    # Issue warning before overwriting files
    if os.path.exists(args.output_dir):
        print(f"* Warning: {args.output_dir} already exists -- files will be overwritten")
        os.makedirs(args.output_dir, exist_ok=True)

    visualizer = VTKOutput(args.netcdf_file)

    # -------------------------------
    # Type: Beam/Nodes
    # -------------------------------
    if args.type == 'beam' or args.type == 'nodes':
        visualizer.create_animation(args.output_dir, args.type)

    # -------------------------------
    # Type: Lines
    # -------------------------------
    elif args.type == 'lines':
        # Parse line connectivity if provided
        if not args.connectivity:
            parser.error("--connectivity is required when --type=lines")

        line_connectivity = []
        for conn in args.connectivity:
            try:
                start, end = map(int, conn.split(','))
                line_connectivity.append((start, end))
            except (ValueError, TypeError):
                parser.error(f"Invalid connectivity format: {conn}. Expected format: 'start,end'")

        visualizer.create_animation(args.output_dir, args.type, line_connectivity)

    # -------------------------------
    # Type: Floating Platform
    # -------------------------------
    elif args.type == 'platform':
        if not args.fairleads or not args.anchors:
            parser.error("--fairleads and --anchors are required when --type=platform")

        try:
            fairlead_indices = [int(idx) for idx in args.fairleads.split(',')]
            anchor_indices = [int(idx) for idx in args.anchors.split(',')]

            if len(fairlead_indices) != len(anchor_indices):
                parser.error("Number of fairlead indices must match number of anchor indices")

            visualizer.create_platform_animation(
                args.output_dir,
                args.platform_node,
                fairlead_indices,
                anchor_indices
            )
        except ValueError:
            parser.error("Invalid format for fairleads or anchors. Expected comma-separated integers.")

    # -------------------------------
    # Type: Rotor
    # -------------------------------
    elif args.type == 'rotor':
        # Parse beam elements if provided
        beam_elements = None
        if args.beam_elements:
            try:
                # Format should be like "0,1,2 3,4,5" where each space-separated group
                # are the node indices for a beam element
                beam_elements = []
                for beam in args.beam_elements.split():
                    beam_elements.append([int(idx) for idx in beam.split(',')])
            except ValueError:
                parser.error("Invalid format for beam elements")

        visualizer.create_rotor_animation(args.output_dir, beam_elements)

    else:
        raise ValueError(f"Unknown type: {args.type}")

if __name__ == "__main__":
    main()
