from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from habitat import EmbodiedTask, registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import DistanceToGoal
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import (
    ActionIDs,
    BaseExplorer,
    BaseExplorerSensorConfig,
)
from frontier_exploration.utils.general_utils import wrap_heading


class State:
    EXPLORE = 0
    BEELINE = 1
    PIVOT = 2


@registry.register_sensor
class ObjNavExplorer(BaseExplorer):
    cls_uuid: str = "objnav_explorer"

    def __init__(
        self,
        sim: HabitatSim,
        config: "DictConfig",
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._beeline_dist_thresh = config.beeline_dist_thresh
        self._success_distance = config.success_distance

        self._episode = None
        self._task = task
        self._state = State.EXPLORE
        self._beeline_target = None
        self._target_yaw = None
        self._episode_view_points: Optional[List[Tuple[float, float, float]]] = None

    def _reset(self, episode):
        super()._reset(episode)
        self._episode = episode
        self._state = State.EXPLORE
        self._beeline_target = None
        self._target_yaw = None
        self._episode_view_points = [
            view_point.agent_state.position
            for goal in episode.goals
            for view_point in goal.view_points
        ]
        self._task.measurements.measures[DistanceToGoal.cls_uuid].reset_metric(episode)

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
        if self._state == State.EXPLORE:
            return super().get_observation(task, episode, *args, **kwargs)

        # Transition to another state if necessary
        min_dist = self._get_min_dist()
        if self._state == State.BEELINE and min_dist < self._success_distance:
            # Change to PIVOT state if already within success distance
            closest_point = self.identify_closest_viewpoint()
            closest_rot = closest_point.agent_state.rotation
            self._target_yaw = 2 * np.arctan2(closest_rot[1], closest_rot[3])
            self._state = State.PIVOT
        elif self._state == State.PIVOT and min_dist > self._success_distance:
            # Change to BEELINE state if we are now out of the success distance
            self._state = State.BEELINE

        # Execute the appropriate behavior for the current state
        if self._state == State.BEELINE:
            self._beeline_target = self._episode._shortest_path_cache.points[-1]
            return self._decide_action(self._beeline_target)
        elif self._state == State.PIVOT:
            return self._pivot()
        else:
            raise ValueError("Invalid state")

    def _pivot(self):
        """Returns LEFT or RIGHT action to pivot the agent towards the target, or STOP
        if the agent is already facing the target as best it can."""
        agent_rot = self._sim.get_agent_state().rotation
        agent_yaw = 2 * np.arctan2(agent_rot.y, agent_rot.w)
        heading_err = -wrap_heading(self._target_yaw - agent_yaw)
        if heading_err > self._turn_angle / 2:
            return ActionIDs.TURN_RIGHT
        elif heading_err < -self._turn_angle / 2:
            return ActionIDs.TURN_LEFT
        return ActionIDs.STOP

    def identify_closest_viewpoint(self):
        """Returns the viewpoint closest to the agent"""
        view_points = [vp for goal in self._episode.goals for vp in goal.view_points]
        min_dist, min_idx = float("inf"), None
        for i, view_point in enumerate(view_points):
            dist = np.linalg.norm(view_point.agent_state.position - self.agent_position)
            if dist < min_dist:
                min_dist, min_idx = dist, i
        return view_points[min_idx]

    def _get_min_dist(self):
        """Returns the minimum distance to the closest target"""
        dist_to_goal = self._task.measurements.measures[DistanceToGoal.cls_uuid]
        dist_to_goal.update_metric(self._episode)
        return dist_to_goal.get_metric()

    def _update_fog_of_war_mask(self):
        updated = (
            super()._update_fog_of_war_mask() if self._state == State.EXPLORE else False
        )

        min_dist = self._get_min_dist()

        if self._state == State.EXPLORE:
            # Start beelining if the minimum distance to the target is less than the
            # set threshold
            if min_dist < self._beeline_dist_thresh:
                self._state = State.BEELINE
                self._beeline_target = self._episode._shortest_path_cache.points[-1]

        return updated


@dataclass
class ObjNavExplorerSensorConfig(BaseExplorerSensorConfig):
    type: str = ObjNavExplorer.__name__
    turn_angle: float = 30.0  # degrees
    beeline_dist_thresh: float = 3  # meters
    success_distance: float = 0.1  # meters


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{ObjNavExplorer.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{ObjNavExplorer.cls_uuid}",
    node=ObjNavExplorerSensorConfig,
)
