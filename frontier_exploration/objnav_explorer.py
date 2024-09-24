from dataclasses import dataclass

import numpy as np
from habitat import registry
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation
from habitat_sim import AgentState
from hydra.core.config_store import ConfigStore

from frontier_exploration.base_explorer import (
    ActionIDs,
)
from frontier_exploration.target_explorer import (
    GreedyExplorerMixin,
    TargetExplorer,
    TargetExplorerSensorConfig,
)
from frontier_exploration.utils.general_utils import wrap_heading


@registry.register_sensor
class ObjNavExplorer(TargetExplorer):
    cls_uuid: str = "objnav_explorer"

    def _setup_pivot(self):
        closest_point = self.identify_closest_viewpoint()
        closest_rot = closest_point.agent_state.rotation
        self._target_yaw = 2 * np.arctan2(closest_rot[1], closest_rot[3])

    def _pivot(self):
        """Returns LEFT or RIGHT action to pivot the agent towards the target,
        or STOP if the agent is already facing the target as best it can."""
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
        if len(self._episode.goals) > 0:
            goals = self._episode.goals
        else:
            goals = self._task._dataset.goals_by_category[  # type: ignore
                self._episode.goals_key
            ]
        if isinstance(goals[0], ObjectGoal):
            view_points = [vp for goal in goals for vp in goal.view_points]
        else:
            agent_state = AgentState(goals[0].position, goals[0].rotation, {})
            view_points = [ObjectViewLocation(agent_state, None)]
        min_dist, min_idx = float("inf"), None
        for i, view_point in enumerate(view_points):
            dist = np.linalg.norm(view_point.agent_state.position - self.agent_position)
            if dist < min_dist:
                min_dist, min_idx = dist, i
        return view_points[min_idx]


@registry.register_sensor
class GreedyObjNavExplorer(GreedyExplorerMixin, ObjNavExplorer):
    cls_uuid: str = "greedy_objnav_explorer"


@dataclass
class ObjNavExplorerSensorConfig(TargetExplorerSensorConfig):
    type: str = ObjNavExplorer.__name__


@dataclass
class GreedyObjNavExplorerSensorConfig(ObjNavExplorerSensorConfig):
    type: str = GreedyObjNavExplorer.__name__


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{ObjNavExplorer.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{ObjNavExplorer.cls_uuid}",
    node=ObjNavExplorerSensorConfig,
)
cs.store(
    package=f"habitat.task.lab_sensors.{GreedyObjNavExplorer.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{GreedyObjNavExplorer.cls_uuid}",
    node=GreedyObjNavExplorerSensorConfig,
)
