import argparse
import numpy as np
import os
import vtk
from netCDF4 import Dataset
from typing import Dict, List, Optional, Tuple

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

        # If no specific nodes requested, use all nodes
        if node_indices is None:
            indices_to_extract = range(self.num_nodes)
        else:
            # Validate that all requested indices are within range
            for idx in node_indices:
                if idx < 0 or idx >= self.num_nodes:
                    raise ValueError(f"Node index {idx} out of range (max: {self.num_nodes-1})")
            indices_to_extract = node_indices

        # Read all data from NetCDF at once for the requested timestep
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


    def _quaternion_to_rotation_matrix(self, quaternion: List[float]) -> np.ndarray:
        """Converts a 4x1 quaternion [w, i, j, k] to a 3x3 rotation matrix.

        The rotation matrix is computed using the following formula:
        R = | 1 - 2(j^2 + k^2)  2(i*j - w*k)  2(i*k + w*j) |
            | 2(i*j + w*k)  1 - 2(i^2 + k^2)  2(j*k - w*i) |
            | 2(i*k - w*j)  2(j*k + w*i)  1 - 2(i^2 + j^2) |

        Args:
            quaternion (list[float]): A list of 4 floats representing the quaternion

        Returns:
            np.ndarray: A 3x3 rotation matrix
        """
        w, i, j, k = quaternion

        ii, jj, kk = i*i, j*j, k*k
        ij, ik, jk = i*j, i*k, j*k
        wi, wj, wk = w*i, w*j, w*k

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

    def _create_vector_array(self, name: str, components: int = 3):
        """Create a VTK double array with the given name and number of components."""
        array = vtk.vtkDoubleArray()
        array.SetNumberOfComponents(components)
        array.SetName(name)
        return array


    def _add_node_data_to_vtk_object(self, vtk_object: vtk.vtkObject, nodes: List[Dict[str, List[float]]]):
        """Add common node data to a VTK object (polydata or unstructured grid).

        Args:
            vtk_object: VTK object to add data to
            nodes: List of node data dictionaries
        """
        # Add orientation data
        orientation_arrays = {
            axis: self._create_vector_array(f"Orientation{axis}")
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
            R = self._quaternion_to_rotation_matrix(quaternion)

            orientation_arrays["X"].InsertNextTuple3(R[0, 0], R[1, 0], R[2, 0])
            orientation_arrays["Y"].InsertNextTuple3(R[0, 1], R[1, 1], R[2, 1])
            orientation_arrays["Z"].InsertNextTuple3(R[0, 2], R[1, 2], R[2, 2])

        for axis in ["X", "Y", "Z"]:
            vtk_object.GetPointData().AddArray(orientation_arrays[axis])

        # Add velocity data
        translation_velocity = self._create_vector_array("TranslationalVelocity")
        rotation_velocity = self._create_vector_array("RotationalVelocity")

        for node in nodes:
            translation_velocity.InsertNextTuple3(*node['velocity'][0:3])
            rotation_velocity.InsertNextTuple3(*node['velocity'][3:6])

        vtk_object.GetPointData().AddArray(translation_velocity)
        vtk_object.GetPointData().AddArray(rotation_velocity)

        # Add acceleration data
        trans_accel = self._create_vector_array("TranslationalAcceleration")
        rot_accel = self._create_vector_array("RotationalAcceleration")

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
        """Enhanced version of _visualize_structure that supports extracting only specific nodes.

        Args:
            timestep (int): Timestep to visualize
            output_path (str): Path to save the output file to
            structure_type (str): Type of structure to visualize ('nodes', 'beam', or 'lines')
            line_connectivity (list[tuple[int, int]], optional): List of node index pairs to connect with lines.
                                                        Required when structure_type is 'lines'.
            node_indices (Optional[List[int]]): If provided, only use these specific node indices.
        """
        # Extract only the requested nodes
        nodes = self._extract_node_data_at_timestep(timestep, node_indices)

        # Create points for the structure
        points = vtk.vtkPoints()
        for node in nodes:
            points.InsertNextPoint(node['position'][0], node['position'][1], node['position'][2])

        if structure_type == 'nodes':
            # Create polydata for nodes
            vtk_object = vtk.vtkPolyData()
            vtk_object.SetPoints(points)

            # Create the writer
            writer = vtk.vtkXMLPolyDataWriter()
            structure_name = "nodes"

        elif structure_type == 'beam':
            # Create unstructured grid for beam
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

        elif structure_type == 'lines':
            print(f"line_connectivity: {line_connectivity}")
            if line_connectivity is None:
                raise ValueError("line_connectivity must be provided when structure_type is 'lines'")

            # Create lines
            lines = vtk.vtkCellArray()
            for start_idx, end_idx in line_connectivity:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, start_idx)
                line.GetPointIds().SetId(1, end_idx)
                lines.InsertNextCell(line)

            # Create polydata for lines
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


    def visualize_floating_platform(
        self,
        timestep: int,
        output_dir: str,
        platform_node_idx: int,
        fairlead_indices: List[int],
        anchor_indices: List[int]
    ):
        """Visualizes a floating platform with mooring lines at a given timestep.

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

        print(f"platform_nodes: {platform_nodes}")

        # Create platform-to-fairleads connections
        platform_connections = []
        for i, _ in enumerate(fairlead_indices):
            # Since we're extracting a subset of nodes, we need to map the original indices
            # The platform node is at index 0, fairleads start at index 1
            # If there are 3 fairleads, there will be 3 connections, and the connections will be:
            # {0, 1}, {0, 2}, {0, 3}
            platform_connections.append((0, i + 1))

        print(f"platform_connections: {platform_connections}")

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

        print(f"mooring_nodes: {mooring_nodes}")

        # Create fairlead-to-anchor connections
        mooring_connections = []
        for i in range(len(fairlead_indices)):
            # Map to the new indices: fairleads start at 0, anchors start at 1 and they alternate
            # i.e. if there are 3 fairleads, there will be 3 anchors, and the connections will be:
            # 0-1, 2-3, 4-5
            mooring_connections.append((2 * i, 2 * i + 1))

        print(f"mooring_connections: {mooring_connections}")

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


    def create_animation(self, output_dir: str, type: str, line_connectivity: Optional[List[Tuple[int, int]]] = None):
        """Creates an animation by generating a series of VTK files at provided timesteps.

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
            self.visualize_floating_platform(
                timestep,
                output_dir,
                platform_node_idx,
                fairlead_indices,
                anchor_indices
            )

        print(f"Generated floating platform animation in {output_dir}")


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

    # input file argument
    parser.add_argument(
        'netcdf_file',
        type=str,
        help='Path to OpenTurbine NetCDF output file e.g. blade_interface.nc'
    )
    # output directory argument
    parser.add_argument(
        '--output_dir',
        type=str,
        default='vtk_output',
        help='Directory for writing the vtk output files'
    )
    # type of visualization argument
    parser.add_argument(
        '--type',
        type=str,
        default='beam',
        help='Type of visualization to generate: beam, nodes, lines, or platform'
    )
    # line connectivity argument
    parser.add_argument(
        '--connectivity',
        type=str,
        nargs='+',
        help='Line connectivity in format "start,end" (space separated list). Required for type=lines'
    )
    # Platform visualization arguments
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
    args = parser.parse_args()

    # Issue warning before overwriting files
    if os.path.exists(args.output_dir):
        print(f"* Warning: {args.output_dir} already exists -- files will be overwritten")
        os.makedirs(args.output_dir, exist_ok=True)

    visualizer = VTKOutput(args.netcdf_file)

    if args.type == 'beam' or args.type == 'nodes':
        visualizer.create_animation(args.output_dir, args.type)

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

    else:
        raise ValueError(f"Unknown type: {args.type}")

if __name__ == "__main__":
    main()
