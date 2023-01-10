from dataclasses import dataclass
from typing import Any

import habitat_sim
import numpy as np
from gym import Space, spaces
from habitat import EmbodiedTask, Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations import fog_of_war, maps
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.explorer import detect_frontier_waypoints

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


@registry.register_sensor
class FrontierWaypoint(Sensor):
    """Returns a waypoint towards the closest frontier"""

    cls_uuid: str = "frontier_waypoint"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._sim = sim

        # Extract information from config
        self._config = config
        self._area_thresh = config.area_thresh
        self._forward_step_size = config.forward_step_size
        self._fov = config.fov
        self._map_resolution = config.map_resolution
        self._success_distance = config.success_distance
        self._turn_angle = np.deg2rad(config.turn_angle)
        self._visibility_dist = config.visibility_dist

        # These public attributes are used by the FrontierExplorationMap measurement
        self.closest_frontier_waypoint = None
        self.top_down_map = None
        self.fog_of_war_mask = None
        self.frontier_waypoints = []

        self._area_thresh_in_pixels = None
        self._visibility_dist_in_pixels = None
        self._agent_position = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=0,
            high=255,
            shape=(1,),
            dtype=np.uint8,
        )

    @property
    def agent_position(self):
        if self._agent_position is None:
            self._agent_position = self._sim.get_agent_state().position
        return self._agent_position

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        if not task.is_episode_active or self.top_down_map is None:
            self._reset_maps()  # New episode, reset maps

        self._update_fog_of_war_mask()
        self.closest_frontier_waypoint = self._get_frontier_waypoint()
        # Decide an action to take
        if self.closest_frontier_waypoint is None:
            next_waypoint = None
        else:
            next_waypoint = self._get_next_waypoint(self.closest_frontier_waypoint)
        return self._decide_action(next_waypoint)

    def _update_fog_of_war_mask(self):
        self.fog_of_war_mask = fog_of_war.reveal_fog_of_war(
            self.top_down_map,
            self.fog_of_war_mask,
            self._get_agent_pixel_coords(),
            self._get_polar_angle(),
            fov=self._fov,
            max_line_len=self._visibility_dist_in_pixels,
        )

    def _get_frontier_waypoint(self):
        # Get waypoint to closest frontier
        frontier_waypoints = detect_frontier_waypoints(
            self.top_down_map,
            self.fog_of_war_mask,
            self._area_thresh_in_pixels,
            xy=self._get_agent_pixel_coords(),
        )
        if len(frontier_waypoints) == 0:
            return None
        # frontiers are in (y, x) format, so we need to do some swapping
        self.frontier_waypoints = frontier_waypoints[:, ::-1]
        closest_frontier_waypoint = self._get_closest_waypoint(
            self.frontier_waypoints, self._get_agent_pixel_coords()
        )
        return closest_frontier_waypoint

    def _get_next_waypoint(self, frontier_waypoint: np.ndarray):
        shortest_path = habitat_sim.nav.ShortestPath()
        shortest_path.requested_start = self.agent_position
        shortest_path.requested_end = self._pixel_to_map_coors(frontier_waypoint)
        assert self._sim.pathfinder.find_path(shortest_path), "Could not find path!"
        next_waypoint = shortest_path.points[1]
        if shortest_path.geodesic_distance < self._success_distance:
            return None
        return next_waypoint

    def _get_closest_waypoint(
        self, waypoints: np.ndarray, agent_position: np.ndarray
    ) -> np.ndarray:
        """A* search to find the closest (geodesic) waypoint to the agent."""
        x0, y0 = agent_position
        euclidean_distances = np.linalg.norm(waypoints - [x0, y0], axis=1)
        sorted_waypoints = waypoints[np.argsort(euclidean_distances)]
        euclidean_distances.sort()
        min_dist = np.inf
        closest_waypoint = None
        for waypoint, heuristic in zip(sorted_waypoints, euclidean_distances):
            if heuristic > min_dist:
                break
            sim_waypoint = self._pixel_to_map_coors(waypoint)
            dist = self._sim.geodesic_distance(agent_position, sim_waypoint)
            if dist < min_dist:
                min_dist = dist
                closest_waypoint = waypoint
        return closest_waypoint

    def _decide_action(self, next_waypoint: np.ndarray) -> np.ndarray:
        if next_waypoint is None:
            return np.array([STOP], dtype=np.int)

        heading_to_waypoint = np.arctan2(
            next_waypoint[2] - self.agent_position[2],
            next_waypoint[0] - self.agent_position[0],
        )
        agent_heading = -self._get_polar_angle() + np.pi / 2.0
        heading_error = wrap_heading(heading_to_waypoint - agent_heading)
        if abs(heading_error) > self._turn_angle:
            if heading_error > 0:
                return np.array([TURN_RIGHT], dtype=np.int)
            else:
                return np.array([TURN_LEFT], dtype=np.int)

        return np.array([MOVE_FORWARD], dtype=np.int)

    def _get_agent_pixel_coords(self) -> np.ndarray:
        a_x, a_y = maps.to_grid(
            self.agent_position[2],
            self.agent_position[0],
            (self.top_down_map.shape[0], self.top_down_map.shape[1]),
            sim=self._sim,
        )
        return np.array([a_x, a_y])

    def _get_polar_angle(self):
        return TopDownMap.get_polar_angle(self)

    def _convert_meters_to_pixel(self, meters: float) -> int:
        return int(
            meters
            / maps.calculate_meters_per_pixel(self._map_resolution, sim=self._sim)
        )

    def _reset_maps(self):
        self._agent_position = None
        self.top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=False,
        )
        self.fog_of_war_mask = np.zeros_like(self.top_down_map)
        self._area_thresh_in_pixels = self._convert_meters_to_pixel(
            self._area_thresh ** 2
        )
        self._visibility_dist_in_pixels = self._convert_meters_to_pixel(
            self._visibility_dist
        )

    def _pixel_to_map_coors(self, pixel: np.ndarray) -> np.ndarray:
        realworld_x, realworld_y = maps.from_grid(
            pixel[0],
            pixel[1],
            (self.top_down_map.shape[0], self.top_down_map.shape[1]),
            self._sim,
        )
        return self._sim.pathfinder.snap_point(
            [realworld_y, self.agent_position[1], realworld_x]
        )


def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi


@dataclass
class FrontierWaypointSensorConfig(LabSensorConfig):
    type: str = FrontierWaypoint.__name__
    # minimum unexplored area (in meters) needed adjacent to a frontier for that
    # frontier to be valid
    area_thresh: float = 3.0  # square meters
    forward_step_size: float = 0.25  # meters
    fov: int = 90
    map_resolution: int = 1024
    success_distance: float = 0.1  # meters
    turn_angle: float = 10.0  # degrees
    visibility_dist: float = 5.0  # in meters


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{FrontierWaypoint.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{FrontierWaypoint.cls_uuid}",
    node=FrontierWaypointSensorConfig,
)
