from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from habitat import EmbodiedTask, registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import (
    BaseExplorer,
    BaseExplorerSensorConfig,
)
from frontier_exploration.utils.path_utils import (
    a_star_search,
    euclidean_heuristic,
    path_dist_cost,
)


@registry.register_sensor
class ObjNavExplorer(BaseExplorer):
    cls_uuid: str = "objnav_explorer"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._beeline_target = None
        self._episode_view_points: Optional[List[Tuple[float, float, float]]] = None
        self.episode = None

    def _reset(self, episode):
        super()._reset(episode)
        self.episode = episode
        self._beeline_target = None
        self._targets = episode.goals
        self._episode_view_points = [
            view_point.agent_state.position
            for goal in episode.goals
            for view_point in goal.view_points
        ]

    @property
    def beeline_target_pixels(self):
        # This property is used by the FrontierExplorationMap measurement
        if self._beeline_target is None:
            return None
        return self._map_coors_to_pixel(self._beeline_target)

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        super()._pre_step(episode)
        if self._beeline_target is None:
            return super().get_observation(task, episode, *args, **kwargs)
        action = self._decide_action(self._beeline_target, stop_at_goal=True)
        return action

    def _update_fog_of_war_mask(self):
        updated = super()._update_fog_of_war_mask()
        # Turn on beelining if any targets have been spotted
        if updated and self._beeline_target is None:
            heuristic_fn = lambda x: euclidean_heuristic(
                np.array(x), self.agent_position
            )
            cost_fn = lambda x: path_dist_cost(x, self.agent_position, self._sim)
            idx, cost = a_star_search(self._episode_view_points, heuristic_fn, cost_fn)
            if cost < self._success_distance:
                self._beeline_target = np.array(self._episode_view_points[idx])
                self.closest_frontier_waypoint = None
        return updated


@dataclass
class ObjNavExplorerSensorConfig(BaseExplorerSensorConfig):
    type: str = ObjNavExplorer.__name__
    turn_angle: float = 30.0  # degrees


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{ObjNavExplorer.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{ObjNavExplorer.cls_uuid}",
    node=ObjNavExplorerSensorConfig,
)
