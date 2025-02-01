import random
from typing import Any

import cv2
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
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war


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
        self._goal_dist_measure: Measure = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ]
        self._step_count: int = 0
        self._should_update_closest_frontier: bool = True
        self._previous_closest_frontier: np.ndarray = np.full(3, np.nan)

    def _reset(self, episode) -> None:
        super()._reset(episode)
        self._state = State.EXPLORE
        self._beeline_target = np.full(3, np.nan)
        self._closest_goal = np.full(3, np.nan)
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

    def _pre_step(self, episode) -> None:
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
                path_cache = self._episode._shortest_path_cache
                self._beeline_target = path_cache.requested_ends[
                    path_cache.closest_end_point_index
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

    def _setup_pivot(self) -> None:
        raise NotImplementedError

    def _pivot(self) -> None:
        raise NotImplementedError

    def _get_min_dist(self):
        """Returns the minimum distance to the closest target"""
        self._goal_dist_measure.update_metric(self._episode, task=self._task)
        dist = self._goal_dist_measure.get_metric()
        path_cache = self._episode._shortest_path_cache
        if path_cache is None or len(path_cache.points) == 0:
            return np.inf

        self._closest_goal = path_cache.requested_ends[
            path_cache.closest_end_point_index
        ]

        return dist

    def _update_fog_of_war_mask(self) -> None:
        updated = (
            super()._update_fog_of_war_mask() if self._state == State.EXPLORE else False
        )

        min_dist = self._get_min_dist()

        if self._state == State.EXPLORE:
            # Start beelining if the minimum distance to the target is less than the
            # set threshold
            if self.check_explored_overlap():
                self._state = State.BEELINE
                path_cache = self._episode._shortest_path_cache
                self._beeline_target = path_cache.requested_ends[
                    path_cache.closest_end_point_index
                ]

        return updated

    def _update_frontiers(self) -> None:
        # There is a small chance that the closest frontier has been filtered out
        # due to self._area_thresh_in_pixels, so we need to run it twice (w/ and w/o
        # filtering) to ensure that the closest frontier is not filtered out.
        orig_thresh = self._area_thresh_in_pixels
        orig_fog = self.fog_of_war_mask.copy()

        self._area_thresh_in_pixels = 0
        super()._update_frontiers()
        self._area_thresh_in_pixels = orig_thresh  # revert to original value

        if len(self.frontier_waypoints) == 0:
            self.fog_of_war_mask = orig_fog  # revert to original value
            super()._update_frontiers()
            return

        # Determine goal frontier when filtering is disabled
        goal_frontier = GreedyExplorerMixin._get_closest_waypoint(self)
        # Determine the corresponding segment
        matches = np.all(self.frontier_waypoints == goal_frontier, axis=1)
        matching_indices = np.where(matches)[0]
        assert len(matching_indices) == 1, (
            f"{len(matching_indices) = }\n"
            f"{self.frontier_waypoints = }\n{goal_frontier = }"
        )
        goal_segment = self._frontier_segments[matching_indices[0]]

        self.fog_of_war_mask = orig_fog  # revert to original value
        super()._update_frontiers()

        # If the goal_frontier is MISSING from the filtered frontiers, add both it and
        # its corresponding segment back
        if len(self.frontier_waypoints) == 0:
            self.frontier_waypoints = np.array([goal_frontier])
            self._frontier_segments = [goal_segment]
        else:
            matches = np.all(self.frontier_waypoints == goal_frontier, axis=1)
            matching_indices = np.where(matches)[0]
            if len(matching_indices) == 0:
                self.frontier_waypoints = np.vstack(
                    [self.frontier_waypoints, goal_frontier]
                )
                self._frontier_segments.append(goal_segment)

    def check_explored_overlap(self) -> bool:
        # Validate inputs
        height, width = self.top_down_map.shape
        y, x = self._map_coors_to_pixel(self._closest_goal)
        assert (
            0 <= y < height and 0 <= x < width
        ), "Coordinate must be within mask bounds"

        # Create mask with longer FOV cone
        longer_fov_mask = reveal_fog_of_war(
            self.top_down_map.copy(),
            self.fog_of_war_mask.copy(),
            self._get_agent_pixel_coords(),
            self.agent_heading,
            fov=self._fov,
            max_line_len=self._visibility_dist_in_pixels
            + self._convert_meters_to_pixel(self._beeline_dist_thresh),
        )

        return longer_fov_mask[y, x] != 0

    def _visualize_map(self) -> np.ndarray:
        vis = super()._visualize_map()

        # Draw the goal point as a filled red circle of size 4
        goal_px = self._map_coors_to_pixel(self._closest_goal)[::-1]
        cv2.circle(vis, tuple(goal_px), 4, (0, 0, 255), -1)

        # Draw the beeline radius circle (not filled) in blue, convert meters to pixels
        beeline_radius = self._convert_meters_to_pixel(self._beeline_dist_thresh)
        cv2.circle(vis, tuple(goal_px), beeline_radius, (255, 0, 0), 1)

        # Draw the success radius circle in green
        success_radius = self._convert_meters_to_pixel(self._config.success_distance)
        cv2.circle(vis, tuple(goal_px), success_radius, (0, 255, 0), 1)

        return vis


class GreedyExplorerMixin:
    def _get_closest_waypoint(self: TargetExplorer):
        """
        Important assumption is made here: the closest goal has not been seen yet
        (i.e., it is not in the fog of war mask). This implies that the agent MUST go
        through a frontier to reach the closest goal.
        """
        # st = time.time()
        if len(self.frontier_waypoints) == 1:
            return self.frontier_waypoints[0]

        # Make a denser version of the shortest path
        sp_3d_interp = GreedyExplorerMixin.interpolate_line(
            self, self._episode._shortest_path_cache.points
        )
        # Convert to pixel coordinates
        sp_px_interp = np.array(
            [self._map_coors_to_pixel(i).astype(np.int32) for i in sp_3d_interp]
        )

        # Find the first intersection with the fog of war mask starting from the goal.
        # Mask needs to be dilated to ensure the path does not just border the mask.
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = self.fog_of_war_mask.copy()
        dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=2)

        sp_px_interp_reversed = sp_px_interp[::-1]
        intersect_idx = find_first_intersection(
            sp_px_interp_reversed, dilated_mask.astype(np.uint8)
        )
        intersect_px = sp_px_interp_reversed[intersect_idx]
        intersect_3d = self._pixel_to_map_coors(intersect_px, snap=False)
        intersect_3d[1] = 0  # don't consider vertical axis

        # Convert from pixel to 3D coordinates simply for higher precision
        f_seg_3d_list = []
        for f_seg in self._frontier_segments:
            f_seg_3d = self._pixel_to_map_coors(f_seg, snap=False)
            f_seg_3d[:, 1] = 0  # don't consider vertical axis
            f_seg_3d_list.append(f_seg_3d)

        # Identify the frontier waypoint whose frontier segment has the closest point to
        # the intersection point
        min_idx = None
        min_dist = np.inf
        for idx, f_seg_3d in enumerate(f_seg_3d_list):
            # Interpolate the line to ensure that the points are not too far apart
            curr_segment = GreedyExplorerMixin.interpolate_line(self, f_seg_3d)

            # Remove any points that appear in other segments
            other_segments = [f for j, f in enumerate(f_seg_3d_list) if j != idx]
            other_points = np.vstack(other_segments).astype(curr_segment.dtype)
            curr_segment = filter_duplicate_rows(other_points, curr_segment)
            if len(curr_segment) == 0:
                continue

            # Use np.linalg.norm to calculate Euclidean distance from each point in the
            # segment to intersect_3d
            dists = np.linalg.norm(curr_segment - intersect_3d, axis=1)
            min_dist_curr = np.min(dists)
            if min_dist_curr < min_dist:
                min_dist = min_dist_curr
                min_idx = idx

        assert (
            min_idx is not None
        ), f"No closest frontier found: {self._scene_id}_{self._episode.episode_id}"

        return self.frontier_waypoints[min_idx]

    def interpolate_line(self: TargetExplorer, line: np.ndarray):
        line = interpolate_line(line, max_dist=0.1)
        line = np.array([self._sim.pathfinder.snap_point(p) for p in line])
        # Filter out any NaNs
        line = line[~np.isnan(line).any(axis=1)]
        return line


def filter_duplicate_rows(array1, array2):
    """
    Remove rows from array2 that are present in array1.

    Parameters:
    array1 : numpy.ndarray of shape (N, 3)
    array2 : numpy.ndarray of shape (M, 3)

    Returns:
    numpy.ndarray : Filtered version of array2 with duplicate rows removed
    """
    # Convert arrays to structured arrays for easier comparison
    # This allows us to treat each row as a single element
    dtype = [("", array1.dtype)] * 3
    struct1 = array1.view(dtype).reshape(-1)
    struct2 = array2.view(dtype).reshape(-1)

    # Create a mask for array2 where True indicates the row is unique
    mask = ~np.in1d(struct2, struct1)

    # Apply the mask to get filtered array2
    return array2[mask]


def interpolate_line(points: np.ndarray, max_dist: float) -> np.ndarray:
    """
    Interpolate additional points along a 3D line such that consecutive points
    are no further than max_dist apart.

    Args:
        points: numpy array of shape (N, 3) containing 3D points
        max_dist: maximum allowed distance between consecutive points

    Returns:
        numpy array of shape (M, 3) containing original and interpolated points
    """
    result = [points[0]]  # Start with first point

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dist = np.linalg.norm(p2 - p1)

        if dist > max_dist:
            # Calculate number of segments needed
            n_segments = int(np.ceil(dist / max_dist))

            # Generate interpolated points
            for j in range(1, n_segments):
                t = j / n_segments
                interp_point = p1 + t * (p2 - p1)
                result.append(interp_point)

        result.append(p2)

    return np.array(result)


def find_closest_point(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Find the point in points1 that has the minimum Euclidean distance to any point in
    points2.

    Args:
        points1: Array of shape (N, 3) containing 3D points
        points2: Array of shape (M, 3) containing 3D points

    Returns:
        The point from points1 that has the smallest distance to any point in points2

    Example:
        >>> p1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        >>> p2 = np.array([[0.1, 0.1, 0.1], [3, 3, 3]])
        >>> find_closest_point(p1, p2)
        array([0, 0, 0])
    """
    # Reshape arrays to enable broadcasting
    # points1_expanded: (N, 1, 3)
    # points2_expanded: (1, M, 3)
    points1_expanded = points1[:, np.newaxis, :]
    points2_expanded = points2[np.newaxis, :, :]

    # Calculate squared distances between all pairs of points
    # Result shape: (N, M)
    squared_distances = np.sum((points1_expanded - points2_expanded) ** 2, axis=2)

    # Find minimum distance for each point in points1
    # min_distances shape: (N,)
    min_distances = np.min(squared_distances, axis=1)

    # Find the index of the point in points1 with the overall minimum distance
    closest_point_idx = np.argmin(min_distances)

    # Return the closest point
    return points1[closest_point_idx]


def find_first_intersection(coords: np.ndarray, mask: np.ndarray) -> int:
    """
    Find the index of the first coordinate that intersects with non-zero values in the
    mask.

    Args:
        coords: np.ndarray of shape (N, 2) containing (y, x) coordinates
        mask: np.ndarray of shape (H, W) containing binary values (0 or 1)

    Returns:
        int: Index of first intersection, or -1 if no intersection found

    Raises:
        ValueError: If coords is not of shape (N, 2) or mask is not 2-dimensional
    """
    # Input validation
    if coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-dimensional, got {mask.ndim} dimensions")

    # Convert coordinates to integers for array indexing
    coords_int = coords.astype(np.int32)

    # Extract y and x coordinates
    y_coords = coords_int[:, 0]
    x_coords = coords_int[:, 1]

    # Get mask values at all coordinates in one operation
    # Clip coordinates to prevent out-of-bounds indexing
    y_clipped = np.clip(y_coords, 0, mask.shape[0] - 1)
    x_clipped = np.clip(x_coords, 0, mask.shape[1] - 1)
    mask_values = mask[y_clipped, x_clipped]

    # Find indices where mask is non-zero
    intersections = np.nonzero(mask_values)[0]

    # Return first intersection index or -1 if none found
    return intersections[0] if len(intersections) > 0 else -1


class TargetExplorerSensorConfig(BaseExplorerSensorConfig):
    turn_angle: float = 30.0  # degrees
    beeline_dist_thresh: float = 8  # meters
    success_distance: float = 0.1  # meters
