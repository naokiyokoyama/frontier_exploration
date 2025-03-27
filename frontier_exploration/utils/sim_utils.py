import os
from typing import List, Optional, Tuple

import habitat_sim
import numpy as np

from frontier_exploration.utils.general_utils import interpolate_path, wrap_heading


def create_simulator(
    scene_path: str,
    agent_height: float = 0.88,
    agent_radius: float = 0.18,
    camera_hfov: float = 79.0,
    camera_position: np.ndarray = np.array([0.0, 0.88, 0.0]),
    camera_resolution: Tuple[int, int] = (480, 640),
    camera_depth_range: Optional[Tuple[float, float]] = None,
) -> habitat_sim.Simulator:
    """
    Create a habitat-sim simulator with the specified parameters.

    Args:
        scene_path (str): Absolute path to the scene 3D asset
        agent_height (float): Height of the agent in meters
        agent_radius (float): Radius of the agent in meters
        camera_hfov (float): Horizontal field of view in degrees
        camera_position (np.ndarray): Camera position relative to agent (x, y, z)
        camera_resolution (Tuple[int, int]): Camera resolution (height, width)
        camera_depth_range (Optional[Tuple[float, float]]): Min and max depth range
            (min, max) for the depth sensor

    Returns:
        habitat_sim.Simulator: Configured habitat-sim simulator
    """
    # Create simulator configuration
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = scene_path
    sim_config.enable_physics = False  # Set to True if physics is needed
    sim_config.allow_sliding = True

    # Create agent configuration
    agent_config = habitat_sim.agent.AgentConfiguration()

    # Configure RGB sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = camera_resolution
    rgb_sensor_spec.position = camera_position
    rgb_sensor_spec.hfov = camera_hfov
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    agent_config.sensor_specifications = [rgb_sensor_spec]

    if camera_depth_range is not None:
        # Configure depth sensor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = camera_resolution
        depth_sensor_spec.position = camera_position
        depth_sensor_spec.hfov = camera_hfov
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        depth_sensor_spec.near = camera_depth_range[0]
        depth_sensor_spec.far = camera_depth_range[1]
        agent_config.sensor_specifications.append(depth_sensor_spec)

    # Create default action space (move_forward, turn_left, turn_right)
    agent_config.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.50)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    # Create simulator configuration and objects
    configuration = habitat_sim.Configuration(sim_config, [agent_config])
    simulator = habitat_sim.Simulator(configuration)

    # Initialize navmesh with agent parameters
    assert simulator.pathfinder.is_loaded, "pathfinder is not loaded!"
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = agent_height
    navmesh_settings.agent_radius = agent_radius
    navmesh_settings.agent_max_climb = 0.10
    navmesh_settings.cell_height = 0.05

    # Generate navmesh
    simulator.recompute_navmesh(
        simulator.pathfinder, navmesh_settings, include_static_objects=False
    )

    # Initialize the agent at a navigable point
    agent = simulator.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = simulator.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)

    return simulator


def get_precise_time() -> float:
    import ntplib

    # Create an NTP client
    ntp_client = ntplib.NTPClient()
    server = ("pool.ntp.org",)  # NTP Pool Project
    # Request time from the server
    response = ntp_client.request(server, timeout=5)

    return response.tx_time


def load_reshaped_npy(filepath: str) -> np.ndarray:
    # Extract original shape from filename
    filename = os.path.basename(filepath)
    shape_str = filename.split("_", 1)[1].rsplit(".", 1)[0]
    orig_shape = tuple(int(i) for i in shape_str.split("_"))

    # Load packed data
    packed = np.load(filepath)

    # Unpack the bits back to original array
    # Calculate total number of elements in original array
    total_elements = np.prod(orig_shape)
    unpacked = np.unpackbits(packed, count=total_elements)

    # Reshape to original dimensions
    fogs = unpacked.reshape(orig_shape).astype(bool)

    return fogs


def generate_path(
    path: np.ndarray,
    degrees_per_step: float = 30,
    max_dist: float = 0.5,
    heading: Optional[float] = None,
) -> List[Tuple[np.ndarray, float]]:
    """
    Given a sparse path of position-only waypoints, generates a dense path with
    interpolated positions and headings, where the first heading is the input heading.

    Args:
        path (np.ndarray): The sparse path of position-only waypoints.
        degrees_per_step (float): How many degrees the agent should rotate in one step
            while turning.
        max_dist (float): How many meters the agent should move in one step while
            moving forward.
        heading (float): The heading of the agent at the start of the path. Random if
            None.


    Returns:
        List[Tuple[np.ndarray, float]]: The dense path with interpolated positions
            and headings.
    """
    if heading is None:
        heading = np.random.uniform(0, 2 * np.pi)
    path = interpolate_path(path, max_dist=max_dist)
    path_with_yaw = add_yaws_to_path(path)  # add headings to the waypoints
    # Replace the heading of the first point of the path with the input heading
    path_with_yaw[0] = path_with_yaw[0][0], wrap_heading(-(heading + np.pi / 2))
    path_with_yaw = interpolate_path_yaw(  # add waypoints between the points with large
        path_with_yaw, degrees_per_step=degrees_per_step  # heading differences
    )
    return path_with_yaw


def add_yaws_to_path(path: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """
    Given a path of positions, adds headings to each position based on the direction to
    the next position from the current position.

    Args:
        path (np.ndarray): The path of positions with shape (N, 3).

    Returns:
        List[Tuple[np.ndarray, float]]: The path with headings added to each position.
    """
    path_with_yaw = []
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        x_diff = end[0] - start[0]
        y_diff = end[2] - start[2]
        yaw = wrap_heading(-(np.arctan2(y_diff, x_diff) + np.pi / 2))
        path_with_yaw.append((start, yaw))
    path_with_yaw.append((path[-1], path_with_yaw[-1][1]))

    return path_with_yaw


def interpolate_path_yaw(
    path_with_yaw: List[Tuple[np.ndarray, float]], degrees_per_step: float
) -> List[Tuple[np.ndarray, float]]:
    """
    Given a path with positions and headings, identifies large differences in headings
    between consecutive waypoints and interpolates between them to add more waypoints
    with smaller heading differences.

    Args:
        path_with_yaw (List[Tuple[np.ndarray, float]]): The path with positions and
            headings.
        degrees_per_step (float): How many degrees the agent should rotate in one step
            while turning.

    Returns:
        List[Tuple[np.ndarray, float]]: The path with interpolated headings.
    """
    large_diff_indices = []
    radians_per_step = np.deg2rad(degrees_per_step)
    for i in range(len(path_with_yaw) - 1):
        yaw_diff = wrap_heading(path_with_yaw[i][1] - path_with_yaw[i + 1][1])
        if abs(yaw_diff) > radians_per_step:
            large_diff_indices.append(i)

    interpolated_path_with_yaw = path_with_yaw.copy()
    for i in large_diff_indices[::-1]:
        (position, start_yaw), (_, end_yaw) = path_with_yaw[i], path_with_yaw[i + 1]
        yaw_diff = abs(wrap_heading(start_yaw - end_yaw))

        num_points = max(1, round(yaw_diff / radians_per_step))
        interpolated_yaws = interpolate_yaws(start_yaw, end_yaw, num_points)
        interpolated_yaws_w_pos = [(position, yaw) for yaw in interpolated_yaws]
        interpolated_path_with_yaw = (
            interpolated_path_with_yaw[: i + 1]
            + interpolated_yaws_w_pos
            + interpolated_path_with_yaw[i + 1 :]  # noqa
        )

    return interpolated_path_with_yaw


def interpolate_yaws(yaw1: float, yaw2: float, num_steps: int) -> List[float]:
    """
    Given two yaw angles, interpolates between them to add more yaw angles with smaller
    differences, excluding the input angles.

    Args:
        yaw1 (float): The first yaw angle.
        yaw2 (float): The second yaw angle.
        num_steps (int): The number of yaw angles to interpolate between the two angles.

    Returns:
        List[float]: The interpolated yaw angles, excluding the input angles.
    """
    # Calculate the shortest rotation between the two yaw angles
    diff_rad = (yaw2 - yaw1 + np.pi) % (2 * np.pi) - np.pi
    if diff_rad < -np.pi:
        diff_rad += 2 * np.pi

    # Generate interpolated yaw angles, excluding the input angles
    interpolated_yaws = []
    for i in range(1, num_steps):
        t = i / num_steps
        interpolated_yaw = yaw1 + t * diff_rad
        interpolated_yaws.append(interpolated_yaw)

    return interpolated_yaws


def get_depth_hfov(sim: habitat_sim.Simulator) -> float:
    """
    Get the horizontal field of view of the depth sensor in the simulator.

    Args:
        sim (habitat_sim.Simulator): The simulator object.

    Returns:
        float: The horizontal field of view of the depth sensor.
    """
    depth_sensor = sim._sensors["depth_sensor"]
    return depth_sensor.specification().hfov


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("scene_path", type=str)
    args = parser.parse_args()

    sim = create_simulator(args.scene_path)
    print("Simulator created")
    d: habitat_sim.simulator.Sensor = sim._sensors["rgb_sensor"]
    print(float(d._spec.hfov))
    print(type(d._spec.hfov))
    sim.close()
    print("Simulator closed")
