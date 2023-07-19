from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat import EmbodiedTask, Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import (
    HeadingSensorConfig,
    LabSensorConfig,
)
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import HeadingSensor
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import BaseExplorer


@registry.register_sensor
class FrontierSensor(Sensor):
    cls_uuid: str = "frontier_sensor"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._curr_ep_id = None
        self.episodic_yaw = 0.0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(1, 3), dtype=np.float32
        )

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Return the 3D coordinates of each frontier."""
        if self._curr_ep_id != episode.episode_id:
            self._curr_ep_id = episode.episode_id
            heading_sensor: HeadingSensor = task.sensor_suite.sensors[  # type: ignore
                "heading"
            ]
            self.episodic_yaw = heading_sensor.get_observation(None, episode)[0]

        explorer: BaseExplorer = task.sensor_suite.sensors[  # type: ignore
            "base_explorer"
        ]

        if len(explorer.frontier_waypoints) == 0:
            return np.zeros((1, 3), dtype=np.float32)

        global_frontiers = explorer._pixel_to_map_coors(explorer.frontier_waypoints)
        global_frontiers = convert_habitat_to_xyz(global_frontiers)

        start_pos = np.array([episode.start_position])
        episode_origin = convert_habitat_to_xyz(start_pos)[0]

        episodic_frontiers = global_to_local(
            global_frontiers, episode_origin, self.episodic_yaw
        )
        episodic_frontiers = flip_xy(episodic_frontiers)

        return episodic_frontiers


def global_to_local(
    global_coors: np.ndarray, local_origin: np.ndarray, local_yaw: float
) -> np.ndarray:
    """Converts global coordinates to local coordinates, given the 3D coordinates of the
    local origin and the yaw of the local frame. Assumes that local frame has 0 pitch
    and roll.

    Args:
        global_coors (np.ndarray): Array of global coordinates to convert
        local_origin (np.ndarray): 3D coordinates of the local origin
        local_yaw (float): Yaw of the local frame

    Returns:
        np.ndarray: Array of local coordinates
    """
    # Shift the global coordinates with respect to the local origin
    # Add a new axis to the local_origin to enable broadcasting over the global_coors
    coors = global_coors - local_origin[np.newaxis, ...]

    # Create the rotation matrix
    # Yaw is rotation around the Z axis. As we are converting from global to local
    # (inverse transform), we take negative of the yaw angle.
    rotation_matrix = np.array(
        [
            [np.cos(-local_yaw), -np.sin(-local_yaw), 0],
            [np.sin(-local_yaw), np.cos(-local_yaw), 0],
            [0, 0, 1],
        ]
    )

    # Apply rotation to every coordinate shifted to the local origin
    # Transpose both matrices to align the dimensions correctly for matmul operation
    local_coors = np.matmul(coors, rotation_matrix.T)

    return local_coors


def convert_habitat_to_xyz(coors: np.ndarray) -> np.ndarray:
    return np.column_stack((-coors[:, 2], coors[:, 0], coors[:, 1])).astype(np.float32)


def flip_xy(coors: np.ndarray) -> np.ndarray:
    pts = np.column_stack((coors[:, 0], -coors[:, 1], coors[:, 2])).astype(np.float32)
    pts = np.column_stack((pts[:, 1], pts[:, 0], pts[:, 2]))
    pts = rotate_z(pts, np.deg2rad(-30))
    return pts


def rotate_z(points, angle):
    """
    Rotate an array of 3D points around the z axis by a given angle in radians.
    """
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points


@dataclass
class FrontierSensorConfig(LabSensorConfig):
    type: str = FrontierSensor.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.lab_sensors.heading_sensor",
    group="habitat/task/lab_sensors",
    name="heading_sensor",
    node=HeadingSensorConfig,
)
cs.store(
    package=f"habitat.task.lab_sensors.{FrontierSensor.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{FrontierSensor.cls_uuid}",
    node=FrontierSensorConfig,
)
