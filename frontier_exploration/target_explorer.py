import random
from typing import Any

import numpy as np
from habitat import EmbodiedTask
from habitat.core.embodied_task import Measure
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import DistanceToGoal
from omegaconf import DictConfig

from frontier_exploration.base_explorer import (
    ActionIDs,
    BaseExplorer,
    BaseExplorerSensorConfig,
)


class State:
    EXPLORE = 0
    BEELINE = 1
    PIVOT = 2
    CANCEL = 3


class TargetExplorer(BaseExplorer):
    def __init__(
        self,
        sim: HabitatSim,
        config: "DictConfig",
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._beeline_dist_thresh: float = config.beeline_dist_thresh
        self._success_distance: float = config.success_distance

        self._task: EmbodiedTask = task
        self._state: int = State.EXPLORE
        self._beeline_target: np.ndarray = np.full(3, np.nan)
        self._goal_dist_measure: Measure = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ]
        self._step_count: int = 0
        self._should_update_closest_frontier: bool = True
        self._previous_closest_frontier: np.ndarray = np.full(3, np.nan)

    def _reset(self, episode):
        super()._reset(episode)
        self._state = State.EXPLORE
        self._beeline_target = np.full(3, np.nan)
        self._goal_dist_measure.reset_metric(episode, task=self._task)
        self._step_count = 0

    @property
    def beeline_target_pixels(self) -> np.ndarray:
        # This property is used by the FrontierExplorationMap measurement
        if np.any(np.isnan(self._beeline_target)):
            return np.full(2, np.nan)
        px_coor = self._map_coors_to_pixel(self._beeline_target)
        a_x, a_y = px_coor
        if a_x < 0 or a_y < 0:
            return np.full(2, np.nan)
        if a_x >= self.top_down_map.shape[0] or a_y >= self.top_down_map.shape[1]:
            return np.full(2, np.nan)
        return px_coor

    def _pre_step(self, episode):
        super()._pre_step(episode)
        if self._episode._shortest_path_cache is None:
            self._goal_dist_measure.reset_metric(episode, task=self._task)
        if np.isinf(self._get_min_dist()):
            print("Invalid episode; goal cannot be reached.")
            self._state = State.CANCEL

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        self._pre_step(episode)
        if self._state == State.EXPLORE:
            action = super().get_observation(task, episode, *args, **kwargs)
            if np.array_equal(action, ActionIDs.STOP):
                # This is undefined behavior. The explorer shouldn't run out of
                # frontiers, but episode is set up incorrectly.
                task.is_stop_called = True
                # Randomly re-assign the action to forward or turn left/right
                action = random.choice(
                    [
                        ActionIDs.MOVE_FORWARD,
                        ActionIDs.TURN_LEFT,
                        ActionIDs.TURN_RIGHT,
                    ]
                )
        elif self._state == State.CANCEL:
            print("STOPPING: Goal is not navigable.")
            action = ActionIDs.STOP
            task.is_stop_called = True
        else:
            # Transition to another state if necessary
            min_dist = self._get_min_dist()
            if self._state == State.BEELINE and min_dist < self._success_distance:
                # Change to PIVOT state if already within success distance
                self._setup_pivot()
                self._state = State.PIVOT
            elif self._state == State.PIVOT and min_dist > self._success_distance:
                # Change to BEELINE state if now out of the success distance
                self._state = State.BEELINE

            # Execute the appropriate behavior for the current state
            if self._state == State.BEELINE:
                pts_cache = self._episode._shortest_path_cache
                self._beeline_target = pts_cache.requested_ends[
                    pts_cache.closest_end_point_index
                ]
                action = self._decide_action(self._beeline_target)
            elif self._state == State.PIVOT:
                action = self._pivot()
            else:
                raise ValueError("Invalid state")

            # Inflection is used by action inflection sensor for IL. This is
            # already done by BaseExplorer when in EXPLORE state.
            if self._prev_action is not None:
                self.inflection = self._prev_action != action
            self._prev_action = action

        self._step_count += 1

        return action

    def _setup_pivot(self):
        raise NotImplementedError

    def _pivot(self):
        raise NotImplementedError

    def _get_min_dist(self):
        """Returns the minimum distance to the closest target"""
        self._goal_dist_measure.update_metric(self._episode, task=self._task)
        dist = self._goal_dist_measure.get_metric()
        if (
            self._episode._shortest_path_cache is None
            or len(self._episode._shortest_path_cache.points) == 0
        ):
            return float("inf")
        return dist

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
                pts_cache = self._episode._shortest_path_cache
                self._beeline_target = pts_cache.requested_ends[
                    pts_cache.closest_end_point_index
                ]

        return updated

    def _update_frontiers(self):
        # There is a small chance that the closest frontier has been filtered out
        # due to self._area_thresh_in_pixels, so we need to run it twice (w/ and w/o
        # filtering) to ensure that the closest frontier is not filtered out.
        orig_frontiers = self.frontier_waypoints.copy()
        orig_thresh = self._area_thresh_in_pixels
        orig_fog = self.fog_of_war_mask.copy()

        self._area_thresh_in_pixels = 0
        super()._update_frontiers()
        self._area_thresh_in_pixels = orig_thresh

        if not self._should_update_closest_frontier:
            self.frontier_waypoints = orig_frontiers
            return

        target_frontier = GreedyExplorerMixin._get_closest_waypoint(self)

        self.fog_of_war_mask = orig_fog
        super()._update_frontiers()

        if not np.any(np.all(self.frontier_waypoints == target_frontier, axis=1)):
            self.frontier_waypoints = np.vstack(
                [target_frontier, self.frontier_waypoints]
            )


class GreedyExplorerMixin:

    def _get_closest_waypoint(self: TargetExplorer):
        if not self._should_update_closest_frontier:
            return self._curr_closest_frontier

        if len(self.frontier_waypoints) == 0:
            return None
        # Identify the closest target object
        self._goal_dist_measure.update_metric(self._episode, task=self._task)
        pts_cache = self._episode._shortest_path_cache
        if pts_cache.closest_end_point_index == -1:
            return None
        closest_point = pts_cache.requested_ends[pts_cache.closest_end_point_index]
        # Identify the frontier waypoint closest to this object
        sim_waypoints = self._pixel_to_map_coors(self.frontier_waypoints)
        idx, _ = self._astar_search(sim_waypoints, start_position=closest_point)
        if idx is None:
            return None

        return self.frontier_waypoints[idx]


class TargetExplorerSensorConfig(BaseExplorerSensorConfig):
    turn_angle: float = 30.0  # degrees
    beeline_dist_thresh: float = 8  # meters
    success_distance: float = 0.1  # meters
