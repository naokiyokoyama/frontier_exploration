from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from habitat import registry
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
from habitat.tasks.nav.nav import TopDownMap, NavigationEpisode
from hydra.core.config_store import ConfigStore

from frontier_exploration.sensors import FrontierWaypoint


@registry.register_measure
class FrontierExplorationMap(TopDownMap):
    def __init__(
        self, sim: "HabitatSim", config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._frontier_exploration_sensor = None
        super().__init__(sim, config, *args, **kwargs)

    def reset_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ) -> None:
        assert "task" in kwargs, "task must be passed to reset_metric!"
        self._frontier_exploration_sensor = kwargs["task"].sensor_suite.sensors[
            FrontierWaypoint.cls_uuid
        ]
        super().reset_metric(episode, *args, **kwargs)

    def get_original_map(self):
        return self._frontier_exploration_sensor.top_down_map.copy()

    def update_fog_of_war_mask(self, agent_position):
        self._fog_of_war_mask = self._frontier_exploration_sensor.fog_of_war_mask.copy()

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        super().update_metric(episode, action, *args, **kwargs)
        # Update the map with visualizations of the frontier waypoints
        new_map = self._metric["map"].copy()
        for waypoint in self._frontier_exploration_sensor.frontier_waypoints:
            if np.allclose(
                waypoint, self._frontier_exploration_sensor.closest_frontier_waypoint
            ):
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(new_map, waypoint, 4, color, -1)
        self._metric["map"] = new_map


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
