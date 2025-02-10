import random
from typing import Any, Optional

import cv2
import habitat_sim
import numpy as np
from habitat import EmbodiedTask
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from omegaconf import DictConfig
from scipy.spatial import cKDTree

from frontier_exploration.base_explorer import (
    ActionIDs,
    BaseExplorer,
    BaseExplorerSensorConfig,
)
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.general_utils import interpolate_path


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
        self._closest_goal: np.ndarray = np.full(3, np.nan)
        self._should_update_closest_frontier: bool = True
        self._valid_goals = np.array([])
        self._valid_path = None
        self._previous_path_start: np.ndarray = np.full(3, np.nan)
        self.greedy_waypoint_idx: int | None = None

    def _reset(self, episode) -> None:
        super()._reset(episode)
        self._state = State.EXPLORE
        self._beeline_target = np.full(3, np.nan)
        self._closest_goal = np.full(3, np.nan)
        if hasattr(self._episode.goals[0], "view_points"):
            # ObjectNav
            self._valid_goals = np.array(
                [
                    view_point.agent_state.position
                    for goal in self._episode.goals
                    for view_point in goal.view_points
                ],
                dtype=np.float32,
            )
        else:
            # ImageNav
            self._valid_goals = np.array(
                [goal.position for goal in self._episode.goals], dtype=np.float32
            )
        self._previous_path_start = np.full(3, np.nan)
        self.filter_multistory_goals()
        self.greedy_waypoint_idx = None

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

    def _pre_step(self, episode) -> None:
        super()._pre_step(episode)
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
                self.filter_multistory_goals()
                self._beeline_target = self._closest_goal
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

        return action

    def _setup_pivot(self) -> np.ndarray:
        raise NotImplementedError

    def _pivot(self) -> np.ndarray:
        raise NotImplementedError

    def _get_min_dist(self):
        """Returns the minimum distance to the closest target"""
        self.filter_multistory_goals()
        self._closest_goal = self._valid_path.requested_ends[
            self._valid_path.closest_end_point_index
        ]
        dist = self._valid_path.geodesic_distance

        return dist

    def _update_fog_of_war_mask(self) -> None:
        updated = (
            super()._update_fog_of_war_mask() if self._state == State.EXPLORE else False
        )

        if self._state == State.EXPLORE:
            # Start beelining if the minimum distance to the target is less than the
            # set threshold
            if self.check_explored_overlap():
                self._state = State.BEELINE
                self._get_min_dist()
                self._beeline_target = self._closest_goal

        return updated

    def update_frontiers(self) -> None:
        # There is a small chance that the closest frontier has been filtered out
        # due to self._area_thresh_in_pixels, so we need to run it twice (w/ and w/o
        # filtering) to ensure that the closest frontier is not filtered out.
        orig_thresh = self._area_thresh_in_pixels
        orig_fog = self.fog_of_war_mask.copy()

        self._area_thresh_in_pixels = 0
        super().update_frontiers()
        self._area_thresh_in_pixels = orig_thresh  # revert to original value

        if len(self.frontier_waypoints) == 0:
            self.fog_of_war_mask = orig_fog  # revert to original value
            super().update_frontiers()
            return

        # Determine goal frontier when filtering is disabled
        goal_frontier = self.frontier_waypoints[
            GreedyExplorerMixin.get_greedy_waypoint_idx(self, adaptive=False)
        ]
        # Determine the corresponding segment
        matches = np.all(self.frontier_waypoints == goal_frontier, axis=1)
        matching_indices = np.where(matches)[0]
        assert len(matching_indices) == 1, (
            f"{len(matching_indices) = }\n"
            f"{self.frontier_waypoints = }\n{goal_frontier = }"
        )
        goal_segment = self._frontier_segments[matching_indices[0]]

        self.fog_of_war_mask = orig_fog  # revert to original value
        super().update_frontiers()

        # If the goal_frontier is MISSING from the filtered frontiers, add both it and
        # its corresponding segment back
        if len(self.frontier_waypoints) == 0:
            self.frontier_waypoints = np.array([goal_frontier])
            self._frontier_segments = [goal_segment]
            self.greedy_waypoint_idx = 0
        else:
            matches = np.all(
                np.isclose(self.frontier_waypoints, goal_frontier, rtol=0, atol=0.75),
                axis=1,
            )
            matching_indices = np.where(matches)[0]
            if len(matching_indices) == 0:
                # Greedy waypoint is missing from the filtered frontiers, add it
                self.frontier_waypoints = np.vstack(
                    [self.frontier_waypoints, goal_frontier]
                )
                self._frontier_segments.append(goal_segment)
                self.greedy_waypoint_idx = len(self.frontier_waypoints) - 1
            else:
                # Greedy waypoint is present in the filtered frontiers, find its index
                assert len(matching_indices) == 1
                self.greedy_waypoint_idx = matching_indices[0]

    def check_explored_overlap(self) -> bool:
        # Validate inputs
        height, width = self.top_down_map.shape
        y, x = self._map_coors_to_pixel(self._closest_goal)
        assert (
            0 <= y < height and 0 <= x < width
        ), "Coordinate must be within mask bounds"

        # Create mask with longer FOV cone
        longer_fov_mask = reveal_fog_of_war(
            top_down_map=self.top_down_map.copy(),
            current_fog_of_war_mask=self.fog_of_war_mask.copy(),
            current_point=self._get_agent_pixel_coords(),
            current_angle=self.agent_heading,
            fov=self._fov,
            max_line_len=self._visibility_dist_in_pixels
            + self._convert_meters_to_pixel(self._beeline_dist_thresh),
        )

        return longer_fov_mask[y, x] != 0

    def filter_multistory_goals(self) -> None:
        if np.array_equal(self.agent_position, self._previous_path_start):
            return

        # Get the shortest path to the goal
        path = None
        while len(self._valid_goals) > 0:
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_ends = self._valid_goals
            path.requested_start = self.agent_position
            self._sim.pathfinder.find_path(path)
            if path.closest_end_point_index == -1:
                # No path found.
                raise RuntimeError("No path found to any goal.")

            if np.ptp(np.array(path.points)[:, 1]) < 0.7:
                break

            # Path is multi-story; remove the goal and all nearby goals from
            # self._valid_goals
            invalid_mask = extract_connected_component(
                points=self._valid_goals,
                target_idx=path.closest_end_point_index,
                dist_thresh=0.2,
            )
            assert np.any(invalid_mask), "No connected component found"
            self._valid_goals = self._valid_goals[~invalid_mask]
            path = None

        assert len(self._valid_goals) > 0
        assert path is not None
        self._valid_path = path
        self._previous_path_start = self.agent_position

    def _visualize_map(self) -> np.ndarray:
        vis = super()._visualize_map()

        # Draw the goal point as a filled red circle of size 4
        goal_px = self._vsf(self._map_coors_to_pixel(self._closest_goal)[::-1])
        cv2.circle(vis, goal_px, 4, (0, 0, 255), -1)

        # Draw the beeline radius circle (not filled) in blue, convert meters to pixels
        beeline_radius = self._convert_meters_to_pixel(self._beeline_dist_thresh)
        cv2.circle(vis, goal_px, self._vsf(beeline_radius), (255, 0, 0), 1)

        # Draw the success radius circle in green
        success_radius = self._convert_meters_to_pixel(self._config.success_distance)
        cv2.circle(vis, goal_px, self._vsf(success_radius), (0, 255, 0), 1)

        return vis


class GreedyExplorerMixin:
    def _get_closest_waypoint_idx(self: TargetExplorer) -> int:
        # For usage when the mixin is inherited
        return GreedyExplorerMixin.get_greedy_waypoint_idx(self)

    def get_greedy_waypoint_idx(self: TargetExplorer, adaptive: bool = False) -> int:
        """
        Important assumption is made here: the closest goal has not been seen yet
        (i.e., it is not in the fog of war mask). This implies that the agent MUST go
        through a frontier to reach the closest goal.
        """
        if len(self.frontier_waypoints) == 1:
            return 0

        self.filter_multistory_goals()

        # Make a denser version of the shortest path; remove vertical coordinate
        sp_2d_interp = interpolate_path(
            np.array(self._valid_path.points)[:, [0, 2]], max_dist=0.1
        )

        # Convert frontier segments from pixel to 3D coordinates
        f_seg_2d_list = [  # remove vertical coordinate
            self._pixel_to_map_coors(f_seg, snap=False)[:, [0, 2]]
            for f_seg in self._frontier_segments
        ]

        # Interpolate and then remove any points that appear in multiple segments to
        # avoid ties
        f_seg_2d_dense_list = []
        for idx, f_seg_2d in enumerate(f_seg_2d_list):
            # Interpolate the line to ensure that the points are not too far apart
            curr_segment = interpolate_path(f_seg_2d, max_dist=0.1)

            # Remove any points that appear in existing segments
            if idx > 0:
                other_segments = [f for j, f in enumerate(f_seg_2d_list) if j < idx]
                other_points = np.vstack(other_segments).astype(curr_segment.dtype)
                curr_segment = filter_duplicate_rows(other_points, curr_segment)
            if len(curr_segment) != 0:
                f_seg_2d_dense_list.append(curr_segment)

        # Store the segments into an array of shape (N, S, 2), where N is the number of
        # segments and S is the number of points in the longest segment. Pad with
        # np.inf so that the calculated minimum Euclidian distance is not affected by
        # the padding.
        max_len = max(len(f_seg) for f_seg in f_seg_2d_dense_list)
        f_seg_2d_dense = np.full((len(f_seg_2d_dense_list), max_len, 2), np.inf)
        for i, f_seg in enumerate(f_seg_2d_dense_list):
            f_seg_2d_dense[i, : len(f_seg)] = f_seg

        # Identify the index of the frontier segment that intersects the trajectory at
        # its latest point within a distance threshold
        min_idx = find_latest_intersecting_sequence(
            f_seg_2d_dense, sp_2d_interp, dist_thresh=0.75, adaptive=adaptive
        )

        assert (
            min_idx is not None
        ), f"No closest frontier found: {self._scene_id}_{self._episode.episode_id}"

        return min_idx


def filter_duplicate_rows(array1, array2):
    """
    Remove rows from array2 that are present in array1.

    Parameters:
    array1 : numpy.ndarray of shape (N, 3)
    array2 : numpy.ndarray of shape (M, 3)

    Returns:
    numpy.ndarray : Filtered version of array2 with duplicate rows removed
    """
    # Convert to contiguous arrays first if needed
    array1 = np.ascontiguousarray(array1)
    array2 = np.ascontiguousarray(array2)

    # Create a structured dtype that combines both coordinates
    dtype = np.dtype([("x", array1.dtype), ("y", array1.dtype)])

    # Convert to structured arrays
    struct1 = array1.view(dtype).reshape(-1)
    struct2 = array2.view(dtype).reshape(-1)

    # Create a mask for array2 where True indicates the row is unique
    mask = ~np.in1d(struct2, struct1)

    # Apply the mask to get filtered array2
    return array2[mask]


def extract_connected_component(
    points: np.ndarray, target_idx: int, dist_thresh: float
) -> np.ndarray:
    """
    Extract the connected component (cluster) containing the target point where points
    are connected if they are within dist_thresh distance of each other.

    Args:
        points: (N, 3) array of 3D points
        target_idx: Index of the target point to find the connected component for
        dist_thresh: Maximum distance threshold for points to be considered connected

    Returns:
        np.ndarray: Boolean mask indicating which points are in the connected component
    """
    # Build KD-tree for efficient nearest neighbor queries
    tree = cKDTree(points)

    # Initialize arrays for tracking visited points and component membership
    n_points = len(points)
    in_component = np.zeros(n_points, dtype=bool)
    to_visit = np.zeros(n_points, dtype=bool)

    # Start with target point
    to_visit[target_idx] = True
    in_component[target_idx] = True

    while np.any(to_visit):
        # Get current point to process
        current_idx = np.where(to_visit)[0][0]
        to_visit[current_idx] = False

        # Find all points within distance threshold
        neighbors = tree.query_ball_point(points[current_idx], dist_thresh)

        # Add unvisited neighbors to the component and queue
        for neighbor_idx in neighbors:
            if not in_component[neighbor_idx]:
                in_component[neighbor_idx] = True
                to_visit[neighbor_idx] = True

    return in_component


def find_latest_intersecting_sequence(
    segments: np.ndarray,
    traj: np.ndarray,
    dist_thresh: float,
    adaptive: bool = True,
) -> Optional[int]:
    """
    Find the sequence in segments that intersects the trajectory at the latest point
    while staying within an adaptive distance threshold.

    Args:
        segments: Array of shape (N, S, 2) containing N sequences of 2D coordinates,
                 each sequence having length S
        traj: Array of shape (M, 2) containing M points of 2D coordinates
        dist_thresh: Base maximum Euclidean distance threshold for considering a point
                    as intersecting. The actual threshold used will be max(dist_thresh,
                    min_distance) where min_distance is the smallest distance between
                    any trajectory point and any sequence point.
        adaptive: Whether to use an adaptive threshold based on the minimum distance
                    between the trajectory and the sequences.

    Returns:
        The index of the sequence that intersects the trajectory at the latest point
        while staying within the adaptive threshold. Returns None if no sequence intersects
        within the threshold.

    Raises:
        ValueError: If input arrays don't match the expected shapes or dimensions
    """
    # Validate input shapes
    if segments.ndim != 3 or segments.shape[-1] != 2:
        raise ValueError(f"segments must have shape (N, S, 2), got {segments.shape}")
    if traj.ndim != 2 or traj.shape[-1] != 2:
        raise ValueError(f"traj must have shape (M, 2), got {traj.shape}")

    N, S, _ = segments.shape
    M, _ = traj.shape

    # Reshape traj to (M, 1, 1, 2) for broadcasting
    traj_expanded = traj[:, np.newaxis, np.newaxis, :]

    # Reshape segments to (1, N, S, 2) for broadcasting
    segments_expanded = segments[np.newaxis, :, :, :]

    # Compute Euclidean distances using broadcasting
    # This creates a temporary array of shape (M, N, S)
    distances = np.sqrt(np.sum((traj_expanded - segments_expanded) ** 2, axis=-1))

    # Find minimum distance along the S dimension
    min_dists = np.min(distances, axis=2)  # Shape: (M, N)

    # Get the global minimum distance
    global_min_dist = np.min(min_dists)

    # Use the maximum of dist_thresh and global_min_dist as the threshold
    if adaptive:
        adaptive_thresh = max(dist_thresh, global_min_dist)
    else:
        adaptive_thresh = dist_thresh

    # Create boolean mask for points within threshold
    intersection_mask = min_dists <= adaptive_thresh  # Shape: (M, N)

    # If no sequences intersect within threshold, return None
    if not np.any(intersection_mask):
        print(f"{global_min_dist = }")
        return None

    # Find the last intersection point for each sequence
    last_intersections = np.where(intersection_mask)[0]  # Get trajectory indices
    sequence_indices = np.where(intersection_mask)[1]  # Get sequence indices

    # Find the sequence with the latest intersection
    latest_idx = np.argmax(last_intersections)
    return sequence_indices[latest_idx]


class TargetExplorerSensorConfig(BaseExplorerSensorConfig):
    turn_angle: float = 30.0  # degrees
    beeline_dist_thresh: float = 8  # meters
    success_distance: float = 0.1  # meters
