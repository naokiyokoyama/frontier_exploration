import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from habitat import EmbodiedTask, registry
from habitat.config import read_write
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
from habitat.tasks.nav.nav import NavigationEpisode, TopDownMap
from habitat.utils.visualizations.utils import observations_to_image
from hydra.core.config_store import ConfigStore

from frontier_exploration.base_explorer import BaseExplorer
from frontier_exploration.objnav_explorer import ObjNavExplorer

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
        if BaseExplorer.cls_uuid in task._config.lab_sensors:
            self._explorer_uuid = BaseExplorer.cls_uuid
        elif ObjNavExplorer.cls_uuid in task._config.lab_sensors:
            self._explorer_uuid = ObjNavExplorer.cls_uuid
        else:
            raise RuntimeError(
                "FrontierExplorationMap needs a BaseExplorer or ObjNavExplorer sensor!"
            )
        explorer_config = task._config.lab_sensors[self._explorer_uuid]
        with read_write(config):
            config.map_resolution = explorer_config.map_resolution
        self._explorer_sensor = None
        super().__init__(sim, config, *args, **kwargs)

    def reset_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ) -> None:
        assert "task" in kwargs, "task must be passed to reset_metric!"
        self._explorer_sensor = kwargs["task"].sensor_suite.sensors[self._explorer_uuid]
        super().reset_metric(episode, *args, **kwargs)

    def get_original_map(self):
        return self._explorer_sensor.top_down_map.copy()

    def update_fog_of_war_mask(self, agent_position):
        self._fog_of_war_mask = self._explorer_sensor.fog_of_war_mask.copy()

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        super().update_metric(episode, action, *args, **kwargs)
        # Update the map with visualizations of the frontier waypoints
        new_map = self._metric["map"].copy()
        circle_size = 20 * self._map_resolution // 1024
        thickness = max(int(round(3 * self._map_resolution / 1024)), 1)
        selected_frontier = self._explorer_sensor.closest_frontier_waypoint
        for waypoint in self._explorer_sensor.frontier_waypoints:
            if selected_frontier is not None and np.array_equal(
                waypoint, selected_frontier
            ):
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)
            cv2.circle(new_map, waypoint[::-1].astype(np.int), circle_size, color, -1)

        beeline_target = getattr(self._explorer_sensor, "beeline_target_pixels", None)
        if beeline_target is not None:
            cv2.circle(
                new_map,
                beeline_target[::-1].astype(np.int),
                circle_size * 2,
                (255, 0, 0),
                thickness * 2,
            )

        next_waypoint = self._explorer_sensor.next_waypoint_pixels
        if next_waypoint is not None:
            cv2.circle(
                new_map,
                next_waypoint[::-1].astype(np.int),
                circle_size,
                (255, 0, 0),
                thickness,
            )
        self._metric["map"] = new_map

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


@dataclass
class FrontierExplorationMapMeasurementConfig(TopDownMapMeasurementConfig):
    type: str = FrontierExplorationMap.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.frontier_exploration_map",
    group="habitat/task/measurements",
    name="frontier_exploration_map",
    node=FrontierExplorationMapMeasurementConfig,
)
