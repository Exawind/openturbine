#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import argparse
import numpy as np
import os
import windIO
import yaml

from scipy.interpolate import PchipInterpolator
from typing import Any, Dict, List, Tuple

#--------------------------------------------------------------------------------
# Helper functions
#--------------------------------------------------------------------------------

def extract_coordinate_data(
    airfoil_info: Dict,
    airfoil_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts x, y coordinates from airfoil info in windIO file.

    Args:
        airfoil_info: Dictionary containing airfoil information
        airfoil_name: Name of airfoil (for error messages)

    Returns:
        Tuple of (x_coords, y_coords) as numpy arrays
    """
    if 'coordinates' not in airfoil_info:
        raise ValueError(f"No coordinate data found for airfoil {airfoil_name}")

    # Capture x, y coordinates from airfoil coordinates dictionary
    airfoil_coords = airfoil_info['coordinates']
    if isinstance(airfoil_coords, dict):
        if 'x' in airfoil_coords and 'y' in airfoil_coords:
            x_original = np.array(airfoil_coords['x'])
            y_original = np.array(airfoil_coords['y'])
            return x_original, y_original

    # Unsupported coordinate format -> error
    raise ValueError(f"Unsupported coordinate format for {airfoil_name}")


def extract_polar_coefficients_data(
    polar_set: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts values of alpha, cl, cd, cm for an airfoil from polar data provided in windIO file.
    Alpha: angle of attack [rad]
    cl: lift coefficient [-]
    cd: drag coefficient [-]
    cm: moment coefficient [-]

    Args:
        polar_set: Dictionary containing polar data

    Returns:
        Tuple of (alpha, cl, cd, cm) numpy arrays
    """
    alpha, cl, cd, cm = None, None, None, None

    if 'c_l' in polar_set and 'c_d' in polar_set:
        alpha = np.array(polar_set['c_l']['grid'])
        cl = np.array(polar_set['c_l']['values'])
        cd = np.array(polar_set['c_d']['values'])
        # cm is optional and might not be present in polar_set
        cm = np.array(polar_set['c_m']['values']) if 'c_m' in polar_set else np.zeros_like(cl)

    if alpha is None or cl is None or cd is None or cm is None:
        available_keys = list(polar_set.keys())
        raise KeyError(f"Could not find all required polar data. Available keys: {available_keys}")

    return alpha, cl, cd, cm

def convert_numpy_array_to_serializable_list(obj: Any) -> Any:
    """
    Converts numpy arrays to lists for YAML serialization.

    Args:
        obj: Object that may contain numpy arrays

    Returns:
        Object with numpy arrays converted to lists
    """
    # numpy arrays -> lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # dictionary -> recursively convert numpy arrays to lists
    if isinstance(obj, dict):
        return {key: convert_numpy_array_to_serializable_list(value) for key, value in obj.items()}
    # list -> recursively convert numpy arrays to lists
    if isinstance(obj, list):
        return [convert_numpy_array_to_serializable_list(item) for item in obj]
    # scalar -> return as is
    return obj

def calculate_arc_length_parameterization(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates cumulative arc length parameterization for a curve.

    Args:
        x: x-coordinates of the curve as numpy array
        y: y-coordinates of the curve as numpy array

    Returns:
        Normalized arc length parameter s in range [0, 1] as numpy array
    """
    dx = np.diff(x) # delta b/w consecutive elements in x
    dy = np.diff(y) # delta b/w consecutive elements in y
    ds = np.sqrt(dx**2 + dy**2) # arc length
    s = np.concatenate([[0], np.cumsum(ds)]) # cumulative arc length
    return s / s[-1]  # normalize -> [0, 1]

#-------------------------------------------------------------------------------
# Core windIO pre-processing logic
#-------------------------------------------------------------------------------

class WindIOPreprocessor:
    """Preprocessor for windIO airfoil and polar data interpolation."""

    def __init__(self, windIO_file: str):
        """
        Initializes the preprocessor with a windIO file and loads all data attributes.

        Args:
            windIO_file: Path to the input windIO YAML file
        """
        self.windIO_file = windIO_file
        self.windIO_data, self.blade_data, self.airfoil_data, self.polar_data = None, None, None, None
        self._initialize_data()

    #-------------------------------------------------------------
    # Initialize data
    #-------------------------------------------------------------

    def _initialize_data(self):
        """Initializes all data attributes.

        Calls the following methods in order:
        - _load_windIO_file -> loads and parses the windIO YAML file
        - _extract_blade_data -> extracts blade and outer_shape_data from windIO structure
        - _extract_airfoil_data -> extracts airfoil data from windIO structure
        - _extract_polar_data -> extracts polar data from airfoil data
        """
        self._load_windIO_file()
        self._extract_blade_data()
        self._extract_airfoil_data()
        self._extract_polar_data()

    def _load_windIO_file(self) -> None:
        """Loads and parses the windIO YAML file."""
        try:
            with open(self.windIO_file, 'r') as file:
                self.windIO_data = yaml.safe_load(file)
            print(f"Successfully loaded windIO file: {self.windIO_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"windIO file not found: {self.windIO_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _extract_blade_data(self) -> None:
        """Extracts blade and outer_shape_data from windIO structure."""
        try:
            self.blade_data = self.windIO_data['components']['blade']
            self.outer_shape_data = self.blade_data['outer_shape_bem']
        except KeyError as e:
            raise KeyError(f"Required blade data not found in windIO file: {e}")

    def _extract_airfoil_data(self) -> None:
        """Extracts airfoil data from windIO structure."""
        data = self.windIO_data['airfoils']
        if data is not None:
            # airfoil data in list format -> convert to dictionary
            if isinstance(data, list):
                airfoil_dict = {}
                for i_airfoil, airfoil in enumerate(data):
                    if isinstance(airfoil, dict):
                        airfoil_name = airfoil.get('name', f'airfoil_{i_airfoil:03d}')
                        airfoil_dict[airfoil_name] = airfoil
                self.airfoil_data = airfoil_dict
                return
            # airfoil data in dictionary format -> use as is
            self.airfoil_data = data
            print("Warning: No airfoil data found in standard locations")

    def _extract_polar_data(self) -> None:
        """Extracts polar data from airfoil data."""
        self.polar_data = {}
        # iterate over airfoils and extract polar data
        for airfoil_name, airfoil_info in self.airfoil_data.items():
            if isinstance(airfoil_info, dict):
                polar_data = None
                if 'polars' in airfoil_info:
                    polar_data = airfoil_info['polars']
                if polar_data is not None:
                    self.polar_data[airfoil_name] = polar_data
        print(f"Found polar data for {len(self.polar_data)} airfoils")

    #-------------------------------------------------------------
    # Interpolate airfoil cross-sections
    #-------------------------------------------------------------

    def _interpolate_airfoil_cross_sections(self, aero_nodes: np.ndarray) -> Dict[str, Any]:
        """
        Interpolates airfoil cross-sections at aerodynamic nodes along the blade span.

        This function:
        - Uses spanwise positions from windIO outer_shape_bem data to locate provided airfoils
        - Normalizes all windIO-provided airfoils to a common grid
        - Interpolates new airfoil cross-section between normalized airfoils using the piecewise
          cubic hermite polynomial (PCHIP) interpolation
        - Interpolates relative thickness (rthick) using PCHIP interpolation

        Args:
            aero_nodes: Array of aerodynamic node positions [0-1] along blade span, r/R

        Returns:
            Dictionary with interpolated airfoil cross-sections at each node, including:
            - airfoil_coordinates: normalized x,y coordinates
            - interpolation_method: 'direct_copy' or 'pchip'
            - source_airfoils: list of provided airfoils used
            - interpolation_weight: weight for interpolation between two airfoils
            - relative_thickness: interpolated thickness ratio (if available)
        """
        # Get spanwise positions and airfoil assignments from outer_shape
        try:
            spanwise_data = self.outer_shape_data['airfoil_position']
            span_positions = np.array(spanwise_data['grid'])
            airfoil_labels = spanwise_data['labels']

            #-------------------------------------------------------------
            # Interpolate relative thickness data (rthick)
            #-------------------------------------------------------------
            rthick_data = None
            if 'rthick' in self.outer_shape_data:
                rthick_positions = np.array(self.outer_shape_data['rthick']['grid'])
                rthick_values = np.array(self.outer_shape_data['rthick']['values'])

                # Create PCHIP interpolator for rthick
                # TODO should we rather use linear interpolation here?
                rthick_interp = PchipInterpolator(rthick_positions, rthick_values)
                rthick_data = {'interpolator': rthick_interp, 'positions': rthick_positions, 'values': rthick_values}

        except KeyError as e:
            raise KeyError(f"Required outer_shape data not found: {e}")

        # Normalize all windIO-provided airfoils to common grid
        normalized_airfoils = {}
        for airfoil_name in set(airfoil_labels):
            try:
                normalized_airfoils[airfoil_name] = self._normalize_airfoil_coordinates(airfoil_name)
                print(f"Normalized airfoil: {airfoil_name}")
            except Exception as e:
                print(f"Warning: Could not normalize airfoil {airfoil_name}: {e}")

        #-------------------------------------------------------------
        # Interpolate airfoil cross-sections @aerodynamic nodes
        #-------------------------------------------------------------
        interpolated_sections = {}
        for i_node, node_pos in enumerate(aero_nodes):
            node_name = f'node_{i_node:03d}'

            # Before first or after last airfoil -> use nearest airfoil as direct copy
            if node_pos <= span_positions[0] or node_pos >= span_positions[-1]:
                airfoil_name = airfoil_labels[0] if node_pos <= span_positions[0] else airfoil_labels[-1]
                if airfoil_name in normalized_airfoils:
                    interpolated_sections[node_name] = {
                        'position': node_pos,
                        'airfoil_coordinates': normalized_airfoils[airfoil_name].copy(),
                        'interpolation_method': 'direct_copy',
                        'source_airfoils': [airfoil_name]
                    }
            # Between airfoils -> interpolate using PCHIP interpolation
            if node_pos > span_positions[0] and node_pos < span_positions[-1]:
                # Find bounding airfoils
                i_upper = np.searchsorted(span_positions, node_pos)
                i_lower = i_upper - 1
                airfoil_lower = airfoil_labels[i_lower]
                airfoil_upper = airfoil_labels[i_upper]

                if airfoil_lower in normalized_airfoils and airfoil_upper in normalized_airfoils:
                    # Calculate interpolation weight based on position of node relative to bounding airfoils
                    weight = (node_pos - span_positions[i_lower]) / (span_positions[i_upper] - span_positions[i_lower])

                    # Interpolate coordinates using PCHIP
                    # TODO if source airfoils are normalized to use same grid, do we need to interpolate x?
                    coords_lower = normalized_airfoils[airfoil_lower]
                    coords_upper = normalized_airfoils[airfoil_upper]
                    x_interpolator = PchipInterpolator([0, 1], [coords_lower['x'], coords_upper['x']], axis=0)
                    y_interpolator = PchipInterpolator([0, 1], [coords_lower['y'], coords_upper['y']], axis=0)

                    interpolated_sections[node_name] = {
                        'position': node_pos,
                        'airfoil_coordinates': {
                            'x': x_interpolator(weight),
                            'y': y_interpolator(weight),
                            's': coords_lower['s'],  # Use same parametrization as source airfoils
                            'num_points': len(x_interpolator(weight))
                        },
                        'interpolation_method': 'pchip',
                        'source_airfoils': [airfoil_lower, airfoil_upper],
                        'interpolation_weight': weight
                    }
                else:
                    print(f"Warning: Could not interpolate at node {node_name} - missing normalized airfoils")

            # Add relative thickness of the airfoil at aero node
            if rthick_data is not None and node_name in interpolated_sections:
                try:
                    rthick_at_node = float(rthick_data['interpolator'](node_pos))
                    interpolated_sections[node_name]['relative_thickness'] = rthick_at_node
                except:
                    print(f"Warning: Could not interpolate relative thickness at {node_name}")

        print(f"Successfully interpolated airfoil sections at all aerodynamic nodes")
        return normalized_airfoils, interpolated_sections

    def _normalize_airfoil_coordinates(self, airfoil_name: str, num_points: int = 300) -> Dict[str, np.ndarray]:
        """
        Normalizes/regularizes airfoil coordinates to a common grid based on normalized surface curve fraction.
        Uses PCHIP interpolation with open trailing edge.

        This normalization/regularization is necessary for blade element analysis because:
        - Different airfoils may have different numbers of coordinate points
        - Original coordinates may have non-uniform spacing along the surface
        - Blade span interpolation i.e. interpolation between two airfoils requires the
          airfoils to be in comparable formats

        The process splits the airfoil into upper/lower surfaces, parameterizes each by arc
        length, then interpolates to a uniform grid (uniform spacing, 300 points by default)
        for consistent analysis.

        Args:
            airfoil_name: Name of the airfoil
            num_points: Number of points for normalized grid

        Returns:
            Dictionary with normalized x, y coordinates
        """
        if airfoil_name not in self.airfoil_data:
            raise ValueError(f"Airfoil {airfoil_name} not found in airfoil data")

        airfoil_info = self.airfoil_data[airfoil_name]
        x_original, y_original = extract_coordinate_data(airfoil_info, airfoil_name)

        # Ensure open trailing edge by checking if first and last points are the same
        if np.allclose([x_original[0], y_original[0]], [x_original[-1], y_original[-1]], atol=1e-6):
            # Remove duplicate trailing edge point
            x_original = x_original[:-1]
            y_original = y_original[:-1]
            print(f"Removed duplicate trailing edge point for {airfoil_name}")

        #-------------------------------------------------------------
        # Split airfoil -> upper/lower surfaces
        #-------------------------------------------------------------
        # Airfoil coordinates convention: TE -> LE -> TE
        # - Starts at trailing edge (TE, x=1, y=~0)
        # - Goes to leading edge (LE, x=0, y=~0) -> upper surface
        # - Goes to trailing edge (TE, x=1, y=~0) -> lower surface

        # Find leading edge -> minimum x coordinate
        leading_edge_idx = np.argmin(x_original)

        # Upper surface: TE -> LE (first half)
        x_upper, y_upper = x_original[:leading_edge_idx+1], y_original[:leading_edge_idx+1]
        # Lower surface: LE -> TE (second half)
        x_lower, y_lower = x_original[leading_edge_idx:], y_original[leading_edge_idx:]

        # Calculate cumulative arc length for each surface
        s_upper = calculate_arc_length_parameterization(x_upper, y_upper) # [0, 1]
        s_lower = calculate_arc_length_parameterization(x_lower, y_lower) # [0, 1]

        #-------------------------------------------------------------
        # Create common normalized grid for airfoil surface
        #-------------------------------------------------------------
        # Upper surface: 0 to 0.5 | Lower surface: 0.5 to 1
        # Create evenly spaced points along each surface
        s_common_upper = np.linspace(0, 0.5, num_points // 2)
        s_common_lower = np.linspace(0.5, 1., num_points // 2)
        # Concatenate upper and lower surface points, avoiding duplicate at 0.5
        s_common = np.concatenate([s_common_upper, s_common_lower[1:]])

        # Interpolate using PCHIP -> upper surface (map to 0-0.5 range)
        x_upper_interp = PchipInterpolator(s_upper * 0.5, x_upper)
        y_upper_interp = PchipInterpolator(s_upper * 0.5, y_upper)

        # Interpolate using PCHIP -> lower surface (map to 0.5-1.0 range)
        x_lower_interp = PchipInterpolator(0.5 + s_lower * 0.5, x_lower)
        y_lower_interp = PchipInterpolator(0.5 + s_lower * 0.5, y_lower)

        # Interpolate to common grid
        x_normalized, y_normalized = np.zeros_like(s_common), np.zeros_like(s_common)

        # mask for upper/lower surfaces and interpolate
        upper_mask = s_common <= 0.5
        x_normalized[upper_mask] = x_upper_interp(s_common[upper_mask])
        y_normalized[upper_mask] = y_upper_interp(s_common[upper_mask])
        lower_mask = s_common > 0.5
        x_normalized[lower_mask] = x_lower_interp(s_common[lower_mask])
        y_normalized[lower_mask] = y_lower_interp(s_common[lower_mask])

        return {
            'x': x_normalized,
            'y': y_normalized,
            's': s_common,
            'num_points': len(s_common)
        }

    #-------------------------------------------------------------
    # Interpolate airfoil polars
    #-------------------------------------------------------------

    def _normalize_all_airfoil_polars(self, target_conditions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Normalizes/regularizes polar data for all available airfoils to a common angle-of-attack grid.

        This function processes all airfoils that have polar data available in the windIO file.
        For each airfoil, it interpolates the polar coefficients (cl, cd, cm) to a common
        angle-of-attack grid using linear interpolation.

        Args:
            target_conditions: Target conditions i.e. angle-of-attack range for interpolation.
            The target conditions typically include a standardized angle-of-attack range
            (e.g., -π to +π radians at 1° resolution) that will be used consistently across
            all airfoils for blade element analysis.

        Returns:
            Dictionary with airfoil names as keys and their normalized polar data as values
        """
        # Process all airfoils that have polar data
        airfoil_polars = {}
        for airfoil_name in self.polar_data.keys():
            try:
                if airfoil_name not in self.polar_data:
                    raise ValueError(f"No polar data found for airfoil: {airfoil_name}")

                # Process the polars for this airfoil
                polar_data = self.polar_data[airfoil_name]
                interpolated_polars = {}
                for _, polar_set in enumerate(polar_data):
                    polar_set_name = f"polars"
                    if isinstance(polar_set, dict):
                        interpolated_polars[polar_set_name] = self._normalize_single_airfoil_polars(
                            polar_set, target_conditions
                        )

                airfoil_polars[airfoil_name] = interpolated_polars
                print(f"Normalized polars for airfoil: {airfoil_name}")

            except Exception as e:
                print(f"Warning: Could not process polars for airfoil {airfoil_name}: {e}")

        return airfoil_polars

    def _normalize_single_airfoil_polars(
        self,
        polar_set: Dict[str, Any],
        target_conditions: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Processes a single airfoil polar set by extracting coefficients and interpolating to target angle-of-attack
        (AoA) conditions using linear interpolation to ensure bounded behavior.

        Args:
            polar_set: Single airfoil polar dataset containing 'alpha', 'cl', 'cd', 'cm'
            target_conditions: Target conditions for interpolation

        Returns:
            Interpolated polar data as dictionary
        """
        try:
            # Extract provided coefficients from polar set
            alpha, cl, cd, cm = extract_polar_coefficients_data(polar_set)
            if not (len(alpha) == len(cl) == len(cd) == len(cm)):
                raise ValueError(f"Inconsistent array lengths: alpha={len(alpha)}, cl={len(cl)}, cd={len(cd)}, cm={len(cm)}")

            # Linearly interpolate each coefficient to target AoA (via numpy.interp)
            target_alpha = target_conditions.get('alpha', alpha)
            cl_interp = np.interp(target_alpha, alpha, cl)
            cd_interp = np.interp(target_alpha, alpha, cd)
            cm_interp = np.interp(target_alpha, alpha, cm)

            # Get Reynolds number if available
            reynolds = polar_set['re'] if 're' in polar_set else 1.e6  # default value

            return {
                'alpha': target_alpha,
                'cl': cl_interp,
                'cd': cd_interp,
                'cm': cm_interp,
                'reynolds': reynolds
            }

        except Exception as e:
            print(f"Error processing polar set: {e}")

    def _interpolate_polars_at_aero_node(
        self,
        node_name: str,
        node_position: float,
        source_airfoils: List[str],
        normalized_polars: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Interpolates polar data at a specific aerodynamic node using pre-normalized data.

        Args:
            node_name: Name of the aerodynamic node
            node_position: Position along blade span, r/R [0-1]
            source_airfoils: List of source airfoil names for interpolation
            normalized_polars: Pre-normalized polar data for all windIO-provided airfoils

        Returns:
            Dictionary containing interpolated polar data for this node
        """
        # Single airfoil source -> direct copy
        if len(source_airfoils) == 1:
            airfoil_name = source_airfoils[0]
            if airfoil_name not in normalized_polars:
                raise ValueError(f"Pre-normalized polar data not found for {airfoil_name}")

            return {
                'position': node_position,
                'interpolation_method': 'direct_copy',
                'source_airfoils': source_airfoils,
                'polar_data': normalized_polars[airfoil_name]
            }

        # Two airfoils source -> interpolate using pchip
        if len(source_airfoils) == 2:
            airfoil_lower, airfoil_upper = source_airfoils
            if missing := [name for name in (airfoil_lower, airfoil_upper) if name not in normalized_polars]:
                raise ValueError(f"Pre-normalized polar data not found for: {missing}")

            interpolated_polars = self._interpolate_polars_via_pchip(normalized_polars, node_position)

            return {
                'position': node_position,
                'interpolation_method': 'pchip',
                'source_airfoils': source_airfoils,
                'interpolation_weight': node_position,
                'polar_data': interpolated_polars
            }

        # More than two airfoils source -> not supported
        raise ValueError(f"Unexpected number of source airfoils for {node_name}: {len(source_airfoils)}")

    # TODO create an alternative approach using linear interpolation
    def _interpolate_polars_via_pchip(self, normalized_polars, weight):
        """
        Interpolates polar data at a specific spanwise position using PCHIP interpolation
        across all available airfoils.

        Args:
            normalized_polars: Dictionary of normalized polar data for all airfoils
            weight: Spanwise position for interpolation [0-1]

        Returns:
            Interpolated polar data
        """
        # Get all available airfoil positions for PCHIP interpolation
        spanwise_data = self.outer_shape_data['airfoil_position']
        all_positions = np.array(spanwise_data['grid'])
        all_labels = spanwise_data['labels']

        # Get all available polar data for this interpolation
        all_polars = {}
        for pos, label in zip(all_positions, all_labels):
            if label in normalized_polars:
                all_polars[pos] = normalized_polars[label]['polars']

        # Stack all coefficient arrays
        positions = list(all_polars.keys())
        cl_arrays = np.array([all_polars[pos]['cl'] for pos in positions])
        cd_arrays = np.array([all_polars[pos]['cd'] for pos in positions])
        cm_arrays = np.array([all_polars[pos]['cm'] for pos in positions])
        reynolds_array = np.array([all_polars[pos]['reynolds'] for pos in positions])

        # Create PCHIP interpolators
        positions_array = np.array(positions)
        cl_interpolator = PchipInterpolator(positions_array, cl_arrays, axis=0)
        cd_interpolator = PchipInterpolator(positions_array, cd_arrays, axis=0)
        cm_interpolator = PchipInterpolator(positions_array, cm_arrays, axis=0)
        reynolds_interpolator = PchipInterpolator(positions_array, reynolds_array)

        # Interpolate at the specific weight i.e. node position on blade span
        interpolated_polar = {
            'alpha': list(all_polars.values())[0]['alpha'],  # Use alpha from any airfoil
            'cl': cl_interpolator(weight),
            'cd': cd_interpolator(weight),
            'cm': cm_interpolator(weight),
            'reynolds': float(reynolds_interpolator(weight))
        }

        return {'polars': interpolated_polar}

    #-------------------------------------------------------------
    # Public methods
    #-------------------------------------------------------------

    def process_all_data(self, aero_nodes: np.ndarray) -> Dict[str, Any]:
        """
        Processes all airfoil cross-sections and polars by
        - normalizing/regularizing the cross-sections and then interpolating the normalized
          cross-sections at each aerodynamic node
        - normalizing/regularizing the provided polars in windIO file to a common angle-of-attack
          grid
        - interpolating the normalized polars at each aerodynamic node

        Args:
            aero_nodes: Aerodynamic node positions

        Returns:
            Complete processed dataset
        """
        # Interpolate airfoil cross-sections
        print("\nInterpolating airfoil cross-sections at aerodynamic nodes...")
        normalized_sections, interpolated_sections = self._interpolate_airfoil_cross_sections(aero_nodes)

        # Create target angle of attack array in radians (from -π to +π radians)
        # TODO: change the target grid to following
        # -pi -> -pi/6 : 1/4 of points
        # -pi/6 -> pi/6 : 1/2 of points
        # pi/6 -> pi : 1/4 of points
        target_alpha = np.linspace(-np.pi, np.pi, 361)  # 361 points for 1° resolution
        target_conditions = {'alpha': target_alpha}

        # Normalize all windIO-provided polar data
        print("\nNormalizing windIO-provided polar data...")
        normalized_polars = self._normalize_all_airfoil_polars(target_conditions)

        # Interpolate polars at each aerodynamic node
        print("\nInterpolating polar data at each aerodynamic node...")
        interpolated_polars = {}
        for node_name, section_data in interpolated_sections.items():
            node_position = section_data['position']
            source_airfoils = section_data['source_airfoils']
            try:
                # Interpolate polars for this node using pre-normalized data
                node_polars = self._interpolate_polars_at_aero_node(
                    node_name, node_position, source_airfoils, normalized_polars
                )
                interpolated_polars[node_name] = node_polars
            except Exception as e:
                print(f"Warning: Could not interpolate polars for {node_name}: {e}")

        print(f"Successfully interpolated polars at all aerodynamic nodes")

        processed_data = {
            'aerodynamic_nodes': aero_nodes.tolist(),
            'normalized_cross_sections': normalized_sections,
            'interpolated_cross_sections': interpolated_sections,
            'normalized_polars': normalized_polars,
            'interpolated_polars': interpolated_polars,
            'metadata': {
                'source_file': self.windIO_file,
                'num_nodes': len(aero_nodes),
                'alpha_range_radians': [-np.pi, np.pi],
                'alpha_points': len(target_alpha),
                'interpolation_method': 'PCHIP',
                'airfoil_normalization': 'surface_curve_fraction',
                'trailing_edge': 'open'
            }
        }

        return processed_data

    def save_processed_data(
        self,
        processed_data: Dict[str, Any],
        output_file: str
    ) -> None:
        """Saves processed data as a new windIO file with additional airfoil information.

        This method creates a new windIO file that includes:
        - All original windIO data
        - Processed airfoil cross-sections at aerodynamic nodes
        - Processed polar data at aerodynamic nodes
        - Metadata about the processing

        Args:
            processed_data: Processed airfoil and polar data
            output_file: Output windIO file path
        """
        new_windio_data = windIO.load_yaml(self.windIO_file) # load original windIO data

        # Replace original airfoils' section and polar data with normalized data
        normalized_sections = processed_data['normalized_cross_sections'] # normalized cross-sections of source data
        normalized_polars = processed_data['normalized_polars'] # normalized polars of source data
        for airfoil in new_windio_data['airfoils']:
            airfoil_name = airfoil.get('name')
            if airfoil_name in normalized_polars:
                # Convert normalized cross-sections to windIO format
                normalized_section_data = normalized_sections[airfoil_name]
                windIO_section = {
                    'name': airfoil_name,
                    'coordinates': {
                        'x': normalized_section_data['x'],
                        'y': normalized_section_data['y']
                    }
                }
                airfoil['coordinates'] = windIO_section['coordinates']

                # Convert normalized polar data to windIO format
                normalized_polar_data = normalized_polars[airfoil_name]
                windIO_polars = []
                for _, polar_set in normalized_polar_data.items():
                    if isinstance(polar_set, dict) and 'alpha' in polar_set:
                        windio_polar = {
                            'configuration': f"Normalized_{airfoil_name}",
                            're': polar_set.get('reynolds', 1000000.0),
                            'c_l': {
                                'grid': polar_set['alpha'],
                                'values': polar_set['cl']
                            },
                            'c_d': {
                                'grid': polar_set['alpha'],
                                'values': polar_set['cd']
                            },
                            'c_m': {
                                'grid': polar_set['alpha'],
                                'values': polar_set['cm']
                            }
                        }
                        windIO_polars.append(windio_polar)

                airfoil['polars'] = windIO_polars
                print(f"Replaced section and polar data for original airfoil: {airfoil_name}")

        # Create processed airfoil entries in the same format as original airfoils
        # Sort by spanwise position first
        sorted_nodes = sorted(
            processed_data['interpolated_cross_sections'].items(),
            key=lambda x: x[1]['position']
        )
        processed_airfoil_names = []
        for node_name, section_data in sorted_nodes:
            # Create airfoil name based on node number
            node_number = node_name.split('_')[1]  # extract "000" from "node_000"
            airfoil_name = f"processed_node_{node_number}"
            processed_airfoil_names.append(airfoil_name)

            # Create airfoil data structure similar to original windIO format
            processed_airfoil = {
                'name': airfoil_name,
                'coordinates': {
                    'x': section_data['airfoil_coordinates']['x'],
                    'y': section_data['airfoil_coordinates']['y']
                },
                'relative_thickness': section_data.get('relative_thickness', 0.0),
                'description': (
                    f"Interpolated airfoil at {section_data['position']:.3f} "
                    f"span position using {section_data['interpolation_method']} method. "
                    f"Source airfoils: {', '.join(section_data['source_airfoils'])}"
                ),
                'polars': []
            }

            # Add polar data if available for this node
            if node_name in processed_data['interpolated_polars']:
                polar_data = processed_data['interpolated_polars'][node_name]
                if 'polar_data' in polar_data:
                    # Convert polar data to the expected windIO format
                    for polar_name, polar_set in polar_data['polar_data'].items():
                        if isinstance(polar_set, dict) and 'alpha' in polar_set:
                            processed_polar = {
                                'configuration': f"Processed_{node_name} using all source airfoil polars",
                                're': polar_set.get('reynolds', 1000000.0),
                                'c_l': {
                                    'grid': polar_set['alpha'],
                                    'values': polar_set['cl']
                                },
                                'c_d': {
                                    'grid': polar_set['alpha'],
                                    'values': polar_set['cd']
                                },
                                'c_m': {
                                    'grid': polar_set['alpha'],
                                    'values': polar_set['cm']
                                }
                            }
                            processed_airfoil['polars'].append(processed_polar)

            new_windio_data['airfoils'].append(processed_airfoil)

        # Update the outer_shape_bem section to include processed airfoils
        if 'components' in new_windio_data and 'blade' in new_windio_data['components']:
            blade_data = new_windio_data['components']['blade']
            if 'outer_shape_bem' in blade_data:
                outer_shape = blade_data['outer_shape_bem']
                if 'airfoil_position' in outer_shape:
                    existing_grid = outer_shape['airfoil_position']['grid']
                    existing_labels = outer_shape['airfoil_position']['labels']

                    new_grid = []
                    new_labels = []
                    new_grid.extend(existing_grid)
                    new_labels.extend(existing_labels)

                    # Add processed airfoil positions (already sorted by position)
                    for node_name, section_data in sorted_nodes:
                        node_position = section_data['position']
                        node_number = node_name.split('_')[1]  # Extract "000" from "node_000"
                        airfoil_name = f"processed_node_{node_number}"

                        new_grid.append(node_position)
                        new_labels.append(airfoil_name)

                    # Sort by position
                    sorted_indices = np.argsort(new_grid)
                    new_grid = [new_grid[i] for i in sorted_indices]
                    new_labels = [new_labels[i] for i in sorted_indices]

                    # Update the outer_shape_bem data
                    outer_shape['airfoil_position']['grid'] = new_grid
                    outer_shape['airfoil_position']['labels'] = new_labels

        # Sort the airfoils section by spanwise position
        # Create a mapping of airfoil names to their positions
        airfoil_positions = {}

        # Get positions for original airfoils from outer_shape_bem
        if 'components' in new_windio_data and 'blade' in new_windio_data['components']:
            blade_data = new_windio_data['components']['blade']
            if 'outer_shape_bem' in blade_data and 'airfoil_position' in blade_data['outer_shape_bem']:
                grid = blade_data['outer_shape_bem']['airfoil_position']['grid']
                labels = blade_data['outer_shape_bem']['airfoil_position']['labels']
                for pos, label in zip(grid, labels):
                    airfoil_positions[label] = pos

        # Sort airfoils by their spanwise position
        sorted_airfoils = sorted(
            new_windio_data['airfoils'],
            key=lambda x: airfoil_positions.get(x.get('name', ''), float('inf'))
        )

        # Replace the airfoils section with sorted version
        new_windio_data['airfoils'] = sorted_airfoils

        # Convert numpy arrays to lists for YAML serialization
        serializable_data = convert_numpy_array_to_serializable_list(new_windio_data)

        try:
            windIO.write_yaml(serializable_data, output_file)
            print(f"Original windIO data with processed airfoils saved to: {output_file}")
            print(f"Added {len(processed_data['interpolated_cross_sections'])} processed airfoils")
        except Exception as e:
            print(f"Error saving windIO file: {e}")

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------

def main():
    """Main function to parse arguments and preprocess windIO data.

    Example usage (run from the build directory):
        python ../src/utilities/scripts/preprocess_windio_for_BE.py \
            tests/regression_tests/interfaces/interfaces_test_files/IEA-15-240-RWT.yaml \
            --output processed_windIO_data.yaml \
            --nodes 50

    NOTE: Output file is overwritten if it already exists.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess windIO airfoil and polar data for Blade Element Analysis"
    )

    #-------------------------------------------------------------
    # Input/output arguments
    #-------------------------------------------------------------

    # windIO input file -- required argument
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to windIO YAML file e.g. IEA-15-240-RWT.yaml",
    )

    # Output file -- optional argument
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="processed_windIO_data.yaml",
        help="Output file path for processed data (default: processed_windIO_data.yaml)",
    )

    #-------------------------------------------------------------
    # Processing parameters
    #-------------------------------------------------------------

    # Number of aerodynamic nodes
    parser.add_argument(
        "--nodes",
        "-n",
        type=int,
        default=50,
        help="Number of linearly spaced aerodynamic nodes along blade span (default: 50)",
    )

    args = parser.parse_args()

    #-------------------------------------------------------------
    # Preprocess windIO data
    #-------------------------------------------------------------

    # Issue warning before overwriting files
    if os.path.exists(args.output):
        print(f"* Warning: {args.output} already exists -- file will be overwritten")

    try:
        preprocessor = WindIOPreprocessor(args.input_file)

        # Process all airfoil cross-sections and polars
        aero_nodes = np.linspace(0., 1., args.nodes)
        processed_data = preprocessor.process_all_data(aero_nodes)
        preprocessor.save_processed_data(processed_data, args.output)

        print(f"Successfully processed windIO data and saved to: {args.output}")

    except Exception as e:
        print(f"Error during preprocessing windIO file: {e}")

if __name__ == "__main__":
    exit(main())
