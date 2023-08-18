import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from habitat import EmbodiedTask, registry
from habitat.config import read_write
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
from habitat.tasks.nav.nav import NavigationEpisode, TopDownMap
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import (
    MAP_INVALID_POINT,
    MAP_SOURCE_POINT_INDICATOR,
    MAP_TARGET_POINT_INDICATOR,
    MAP_VALID_POINT,
    MAP_VIEW_POINT_INDICATOR,
)
from habitat.utils.visualizations.utils import observations_to_image
from hydra.core.config_store import ConfigStore

from frontier_exploration.base_explorer import BaseExplorer
from frontier_exploration.objnav_explorer import GreedyObjNavExplorer, ObjNavExplorer

DEBUG = os.environ.get("MAP_DEBUG", "False").lower() == "true"
if DEBUG:
    print(f"[{os.path.basename(__file__)}]: WARNING: MAP_DEBUG is True")


@registry.register_measure
class FrontierExplorationMap(TopDownMap):
    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        self._explorer_uuid = None
        for i in [BaseExplorer, ObjNavExplorer, GreedyObjNavExplorer]:
            if i.cls_uuid in task._config.lab_sensors:
                assert (
                    self._explorer_uuid is None
                ), "FrontierExplorationMap only supports 1 explorer sensor at a time!"
                self._explorer_uuid = i.cls_uuid

        if self._explorer_uuid is None:
            raise RuntimeError("FrontierExplorationMap needs an exploration sensor!")
        explorer_config = task._config.lab_sensors[self._explorer_uuid]
        with read_write(config):
            config.map_resolution = explorer_config.map_resolution

        super().__init__(sim, config, *args, **kwargs)

        self._explorer_sensor = None
        self._draw_waypoints: bool = config.draw_waypoints
        self._is_feasible: bool = True

    def reset_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ) -> None:
        assert "task" in kwargs, "task must be passed to reset_metric!"
        self._explorer_sensor = kwargs["task"].sensor_suite.sensors[self._explorer_uuid]
        super().reset_metric(episode, *args, **kwargs)

    def get_original_map(self):
        return self._explorer_sensor.top_down_map.copy()

    def update_fog_of_war_mask(self, *args, **kwargs):
        self._fog_of_war_mask = self._explorer_sensor.fog_of_war_mask.copy()

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        super().update_metric(episode, action, *args, **kwargs)
        # Update the map with visualizations of the frontier waypoints
        new_map = self._metric["map"].copy()
        circle_size = 20 * self._map_resolution // 1024
        thickness = max(int(round(3 * self._map_resolution / 1024)), 1)
        selected_frontier = self._explorer_sensor.closest_frontier_waypoint

        if self._draw_waypoints:
            next_waypoint = self._explorer_sensor.next_waypoint_pixels
            if next_waypoint is not None:
                cv2.circle(
                    new_map,
                    tuple(next_waypoint[::-1].astype(np.int)),
                    circle_size,
                    MAP_INVALID_POINT,
                    1,
                )

        for waypoint in self._explorer_sensor.frontier_waypoints:
            if np.array_equal(waypoint, selected_frontier):
                color = MAP_TARGET_POINT_INDICATOR
            else:
                color = MAP_SOURCE_POINT_INDICATOR
            cv2.circle(
                new_map,
                waypoint[::-1].astype(np.int),
                circle_size,
                color,
                1,
            )

        beeline_target = getattr(self._explorer_sensor, "beeline_target_pixels", None)
        if beeline_target is not None:
            cv2.circle(
                new_map,
                tuple(beeline_target[::-1].astype(np.int)),
                circle_size * 2,
                MAP_SOURCE_POINT_INDICATOR,
                thickness,
            )
        self._metric["map"] = new_map
        self._metric["is_feasible"] = self._is_feasible

        if DEBUG:
            import time

            if not os.path.exists("map_debug"):
                os.mkdir("map_debug")
            img = observations_to_image(
                {}, {f"top_down_map.{k}": v for k, v in self._metric.items()}
            )
            cv2.imwrite(
                f"map_debug/{int(time.time())}_full.png",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

    def _draw_goals_view_points(self, episode):
        super()._draw_goals_view_points(episode)

        # Use this opportunity to determine whether this episode is feasible to complete
        # without climbing stairs

        # Compute the pixel location of the start position
        t_x, t_y = maps.to_grid(
            episode.start_position[2],
            episode.start_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # The start position would be here: self._top_down_map[t_x,t_y]

        # Compute contours that contain MAP_VALID_POINT and/or MAP_VIEW_POINT_INDICATOR
        valid_with_viewpoints = self._top_down_map.copy()
        valid_with_viewpoints[
            valid_with_viewpoints == MAP_VIEW_POINT_INDICATOR
        ] = MAP_VALID_POINT
        contours, _ = cv2.findContours(
            valid_with_viewpoints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # For each contour, draw a filled mask, and then check if the start position is
        # in the mask. If it is: if the mask for this contour has both MAP_VALID_POINT
        # and MAP_VIEW_POINT_INDICATOR in self._top_down_map, then this episode is
        # feasible. Otherwise, it is not.
        is_feasible = False
        for c in contours:
            mask = np.zeros_like(valid_with_viewpoints)
            mask = cv2.drawContours(mask, [c], 0, 1, -1)

            # Check if the start position is in the mask
            if mask[t_x, t_y] == 1:
                masked_values = self._top_down_map[mask.astype(np.bool)]
                values = set(masked_values.tolist())
                is_feasible = (
                    MAP_VALID_POINT in values and MAP_VIEW_POINT_INDICATOR in values
                )
                break

        self._is_feasible = is_feasible


@dataclass
class FrontierExplorationMapMeasurementConfig(TopDownMapMeasurementConfig):
    type: str = FrontierExplorationMap.__name__
    draw_waypoints: bool = True


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.frontier_exploration_map",
    group="habitat/task/measurements",
    name="frontier_exploration_map",
    node=FrontierExplorationMapMeasurementConfig,
)
