from __future__ import annotations

import glob
import json
import os
import os.path as osp
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import cv2
import habitat_sim
import numpy as np
import quaternion as qt
from habitat import EmbodiedTask, registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import NavigationEpisode
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import ActionIDs, BaseExplorer, get_polar_angle
from frontier_exploration.objnav_explorer import ObjNavExplorer
from frontier_exploration.target_explorer import (
    State,
    TargetExplorer,
    TargetExplorerSensorConfig,
    TargetFrontierException,
)
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.frontier_filtering import (
    FrontierFilter,
    FrontierFilterData,
)
from frontier_exploration.utils.general_utils import (
    images_to_video,
    interpolate_path,
    wrap_heading,
)
from frontier_exploration.utils.path_utils import get_path
from frontier_exploration.utils.viz import (
    add_text_to_image,
    add_translucent_border,
    get_mask_except_nearest_contour,
    pad_images_to_max_dim,
    place_image_centered,
    resize_image_maintain_ratio,
    rotate_image_orientation,
    tile_images,
)

EXPLORATION_THRESHOLD = 0.1


def default_on_exception(default_value):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if os.environ.get("EXP_DEBUG") == "1" and not isinstance(
                    e, TargetFrontierException
                ):
                    raise e
                print(f"Exception occurred: {e}")
                if os.environ.get("NO_TRACEBACK") != "1":
                    print("Full traceback:")
                    traceback.print_exc()
                print(
                    f"Exception occurred in episode {args[0]._episode.episode_id} in"
                    f" scene {args[0]._scene_id}"
                )
                return default_value

        return wrapper

    return decorator


@registry.register_sensor
class ExplorationEpisodeGenerator(TargetExplorer):
    cls_uuid: str = "exploration_episode_generator"

    def __init__(
        self,
        sim: HabitatSim,
        config: "DictConfig",
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(sim, config, task, *args, **kwargs)
        self._dataset_path = config.dataset_path
        self._max_exploration_attempts: int = config.max_exploration_attempts
        self._min_exploration_steps: int = config.min_exploration_steps
        self._max_exploration_steps: int = config.max_exploration_steps
        self._min_exploration_coverage: float = config.min_exploration_coverage
        self._max_exploration_coverage: float = config.max_exploration_coverage
        self._map_measure = task.measurements.measures["top_down_map"]
        self._unique_episodes_only = config.unique_episodes_only
        self._task_type = config.task_type
        self._stop_at_beelining: bool = config.stop_at_beelining
        assert config.task_type in ["objectnav", "imagenav"]

        self._is_exploring: bool = False

        self.frontier_filter = FrontierFilter(
            self._fov, int(self._visibility_dist_in_pixels * 1.5)
        )
        self._bad_idx_to_good_idx: Dict[int, int] = {}

        # Fields for storing data that will be recorded into the dataset
        self._exploration_poses: list[list[float]] = []
        self._exploration_imgs: list[np.ndarray] = []
        self._exploration_fogs: list[np.ndarray] = []

        self._gt_fog_of_war_mask: np.ndarray = np.empty((1, 1))  # 2D array
        self._latest_fog: np.ndarray = np.empty((1, 1))  # 2D array
        self._gt_traj: List[GTTrajectoryState] = []
        self._exploration_successful: bool = False
        self._start_z: float = -1.0
        self._step_count: int = 0
        self._timestep_to_greedy_idx: Dict[int, int] = {}

        # For ImageNav
        self._imagenav_goal: np.ndarray = np.empty((1, 1))  # 2D array
        self._total_fov_pixels: int = 0

        # This will just be used for debugging
        self._bad_episode: bool = False
        self._viz_imgs: list[np.ndarray] = []

        self._f_seg_to_3d = {}
        self._previous_f_segs = {}
        self._f_seg_index_to_f_dist = defaultdict(dict)
        self._f_goal_distance_cache = defaultdict(dict)
        self._agent_distance_cache = defaultdict(dict)
        self._f_seg_pose_to_goal_paths = defaultdict(dict)
        self._all_frontier_paths: List[
            Tuple[habitat_sim.MultiGoalShortestPath, habitat_sim.ShortestPath]
        ] = []

    def _reset(self, episode: NavigationEpisode) -> None:
        super()._reset(episode)

        self._visibility_dist = self._config.visibility_dist
        self._area_thresh = self._config.area_thresh

        self._is_exploring = False
        self._minimize_time = False

        self.frontier_filter.reset()
        self._bad_idx_to_good_idx = {}

        self._gt_fog_of_war_mask = None
        self._latest_fog = None
        self._gt_traj = []
        self._exploration_poses = []
        self._exploration_imgs = []
        self._exploration_successful = False
        self._coverage_masks = []
        self._step_count = 0
        self._timestep_to_greedy_idx = {}

        self._f_seg_to_3d = {}
        self._previous_f_segs = {}
        self._f_seg_index_to_f_dist = defaultdict(dict)
        self._f_goal_distance_cache = defaultdict(dict)
        self._agent_distance_cache = defaultdict(dict)
        self._f_seg_pose_to_goal_paths = defaultdict(dict)
        self._all_frontier_paths = []

        self._viz_imgs = []

        if self._task_type == "imagenav":
            self._total_fov_pixels = 0
            self._imagenav_goal = self._generate_imagenav_goal(episode)

        # If the last episode failed, then we need to record the episode's id and its
        # scene id for further debugging
        assert hasattr(episode, "episode_id") and hasattr(
            episode, "scene_id"
        ), "Episode must have episode_id and scene_id attributes"
        if self._bad_episode:
            # Create a blank file in the cwd with the episode id and scene id
            print(f"Episode {self._scene_id} {episode.episode_id} failed!!")
            # filename = f"{episode.episode_id}_{self._scene_id}.txt"
            # with open(filename, "w") as f:
            #     f.write("")
        elif self._step_count > 0:
            print(f"Episode {self._scene_id} {episode.episode_id} succeeded!!")

        self._bad_episode = False
        # Explicitly convert from numpy.float64 to float to avoid JSON serialization
        # issues
        self._start_z = self._sim.get_agent_state().position[1].item()

        print(f"Starting episode {episode.episode_id} in scene {self._scene_id}")

    @property
    def greedy_frontier_idx(self) -> int:
        return self._timestep_to_greedy_idx[self._step_count]

    def _record_frontiers(self) -> None:
        gt_idx = self.greedy_frontier_idx if len(self._frontier_segments) > 0 else -1

        result: FrontierFilterData = self.frontier_filter.score_and_filter_frontiers(
            curr_f_segments=self._frontier_segments,
            curr_cam_pos=self._get_agent_pixel_coords(),
            curr_cam_yaw=self.agent_heading,
            top_down_map=self.top_down_map,
            curr_timestep_id=self._step_count,
            gt_idx=gt_idx,
            filter=True,
            return_all=True,
        )

        self._bad_idx_to_good_idx = result.filtered.bad_idx_to_good_idx

        correct_frontier, correct_frontier_unscored = [
            i.good_indices_to_timestep[gt_idx]
            if len(self._frontier_segments) > 0
            else -1
            for i in (result.filtered, result.unscored_filtered)
        ]

        self._all_frontier_paths = self._get_frontier_dtgs()
        all_frontier_dtgs = [
            i.geodesic_distance + j.geodesic_distance
            for i, j in self._all_frontier_paths
        ]
        frontier_dtgs = {}
        for k, ftd in (
            ("dtgs", result.unfiltered),
            ("dtgs_unscored", result.unscored_unfiltered),
        ):
            frontier_dtgs[k] = {
                timestep: all_frontier_dtgs[idx]
                for idx, timestep in ftd.good_indices_to_timestep.items()
            }

        pose = tuple(float(i) for i in (*self.agent_position, self.agent_heading))
        self._gt_traj.append(
            GTTrajectoryState(
                timestep_id=self._step_count,
                pose=pose,  # noqa
                rgb=self._sim.get_observations_at()["rgb"],
                all_frontiers=list(result.filtered.good_indices_to_timestep.values()),
                all_frontiers_unfiltered=list(
                    set(result.unfiltered.good_indices_to_timestep.values())
                ),
                all_frontiers_unscored=list(
                    result.unscored_filtered.good_indices_to_timestep.values()
                ),
                all_frontiers_unscored_unfiltered=list(
                    result.unscored_unfiltered.good_indices_to_timestep.values()
                ),
                correct_frontier=correct_frontier,
                correct_frontier_unscored=correct_frontier_unscored,
                single_fog_of_war=self._latest_fog,
                distance_to_goal=self._valid_path.geodesic_distance,
                frontier_dtgs=frontier_dtgs,
            )
        )

        assert len(self._gt_traj) - 1 == self._step_count

    def _get_frontier_dtgs(
        self,
    ) -> List[Tuple[habitat_sim.MultiGoalShortestPath, habitat_sim.ShortestPath]]:
        if len(self._frontier_segments) == 0:
            return []

        goal_paths: List[
            Tuple[habitat_sim.MultiGoalShortestPath, habitat_sim.ShortestPath]
        ] = []
        pose_key = tuple(map(float, self.agent_position.tolist()))
        all_f_seg_keys = [tuple(f_seg.flatten()) for f_seg in self._frontier_segments]
        all_pt_keys = []
        for f_seg, f_seg_key in zip(self._frontier_segments, all_f_seg_keys):
            # If the minimum distance to the frontier has already been calculated, use
            # that value
            if pose_key in self._f_seg_pose_to_goal_paths[f_seg_key]:
                goal_paths.append(self._f_seg_pose_to_goal_paths[f_seg_key][pose_key])
                continue

            # If we already interpolated the frontier segment, use that
            if f_seg_key in self._f_seg_to_3d:
                f_seg_3d = self._f_seg_to_3d[f_seg_key]
            else:
                f_seg_3d = interpolate_path(
                    np.array([self._pixel_to_map_coors(i) for i in f_seg]), max_dist=0.3
                )
                self._f_seg_to_3d[f_seg_key] = f_seg_3d

            # See if the point on the frontier closest to the agent has changed
            agent_to_frontier = habitat_sim.MultiGoalShortestPath()
            agent_to_frontier.requested_ends = f_seg_3d
            agent_to_frontier.requested_start = self.agent_position
            path_found = self._sim.pathfinder.find_path(agent_to_frontier)
            assert path_found
            closest_idx = agent_to_frontier.closest_end_point_index

            # If it hasn't changed, then use the previously calculated minimum
            # distance between the frontier and the goals
            if closest_idx in self._f_seg_index_to_f_dist[f_seg_key]:
                frontier_to_goal, f_pt = self._f_seg_index_to_f_dist[f_seg_key][
                    closest_idx
                ]
                agent_to_frontier_goal = habitat_sim.ShortestPath()
                agent_to_frontier_goal.requested_start = self.agent_position
                agent_to_frontier_goal.requested_end = f_pt
                assert self._sim.pathfinder.find_path(agent_to_frontier_goal)
                goal_paths.append((frontier_to_goal, agent_to_frontier_goal))
                self._f_seg_pose_to_goal_paths[f_seg_key][pose_key] = (
                    frontier_to_goal,
                    agent_to_frontier_goal,
                )
                assert len(agent_to_frontier_goal.points) > 0
                continue

            min_dist = np.inf
            best_paths = None
            for pt in f_seg_3d:
                # Get or compute path to goals
                pt_key = tuple(pt)
                all_pt_keys.append(pt_key)
                valid_goals_key = tuple(self._valid_goals.flatten())
                if valid_goals_key in self._f_goal_distance_cache[pt_key]:
                    frontier_to_goal = self._f_goal_distance_cache[pt_key][
                        valid_goals_key
                    ]
                else:
                    frontier_to_goal = habitat_sim.MultiGoalShortestPath()
                    frontier_to_goal.requested_start = pt
                    frontier_to_goal.requested_ends = self._valid_goals
                    self._sim.pathfinder.find_path(frontier_to_goal)
                    self._f_goal_distance_cache[pt_key][
                        valid_goals_key
                    ] = frontier_to_goal

                if pose_key in self._agent_distance_cache[pt_key]:
                    agent_to_frontier_goal = self._agent_distance_cache[pt_key][
                        pose_key
                    ]
                else:
                    agent_to_frontier_goal = habitat_sim.ShortestPath()
                    agent_to_frontier_goal.requested_start = self.agent_position
                    agent_to_frontier_goal.requested_end = pt
                    self._sim.pathfinder.find_path(agent_to_frontier_goal)
                    self._agent_distance_cache[pt_key][
                        pose_key
                    ] = agent_to_frontier_goal

                total_dist = (
                    frontier_to_goal.geodesic_distance
                    + agent_to_frontier_goal.geodesic_distance
                )
                if total_dist < min_dist:
                    min_dist = total_dist
                    self._f_seg_index_to_f_dist[f_seg_key][closest_idx] = (
                        frontier_to_goal,
                        pt,
                    )
                    best_paths = (frontier_to_goal, agent_to_frontier_goal)

            assert best_paths is not None
            goal_paths.append(best_paths)
            assert len(best_paths[1].points) > 0
            self._f_seg_pose_to_goal_paths[f_seg_key][pose_key] = best_paths

        self._f_seg_pose_to_goal_paths = defaultdict(
            dict, {k: self._f_seg_pose_to_goal_paths[k] for k in all_f_seg_keys}
        )
        self._f_seg_to_3d = defaultdict(
            dict, {k: self._f_seg_to_3d[k] for k in all_f_seg_keys}
        )
        self._f_seg_index_to_f_dist = defaultdict(
            dict, {k: self._f_seg_index_to_f_dist[k] for k in all_f_seg_keys}
        )
        self._f_goal_distance_cache = defaultdict(
            dict, {k: self._f_goal_distance_cache[k] for k in all_pt_keys}
        )

        return goal_paths

    def _generate_imagenav_goal(self, episode: NavigationEpisode) -> np.ndarray:
        """
        Generates an image taken at the goal point for ImageNav tasks. The yaw of the
        agent when the image is taken is selected such that at least 50% of the agent's
        FOV is unobstructed. This is to avoid image goals that are too close to walls.

        Args:
            episode (NavigationEpisode): The current episode.

        Returns:
            np.ndarray: The RGB image of the goal.
        """
        if self._total_fov_pixels == 0:
            height, width = self.top_down_map.shape
            single_fov = reveal_fog_of_war(
                top_down_map=np.ones_like(self.top_down_map),  # fully navigable map
                current_fog_of_war_mask=np.zeros_like(
                    self.fog_of_war_mask
                ),  # no explored areas
                current_point=np.array((height // 2, width // 2)),
                current_angle=0.0,
                fov=self._fov,
                max_line_len=self._visibility_dist_in_pixels,
            )

            self._total_fov_pixels = np.count_nonzero(single_fov)

        assert len(episode.goals) == 1, "Only one goal is supported for ImageNav tasks"
        goal_position = self._map_coors_to_pixel(episode.goals[0].position)
        for _ in range(50):
            rand_angle = np.random.uniform(0, 2 * np.pi)
            goal_rotation = qt.quaternion(
                np.cos(rand_angle / 2), 0, np.sin(rand_angle / 2), 0
            )
            yaw = get_polar_angle(goal_rotation)

            curr_fov = reveal_fog_of_war(
                top_down_map=self.top_down_map,
                current_fog_of_war_mask=np.zeros_like(
                    self.fog_of_war_mask
                ),  # no explored areas
                current_point=goal_position,
                current_angle=yaw,
                fov=self._fov,
                max_line_len=self._visibility_dist_in_pixels,
            )
            current_fov_pixels = np.count_nonzero(curr_fov)
            unobstructed_percentage = current_fov_pixels / self._total_fov_pixels

            if unobstructed_percentage > 0.2:
                image_goal = self._sim.get_observations_at(
                    position=episode.goals[0].position,
                    rotation=goal_rotation,
                    keep_agent_at_new_pose=False,
                )["rgb"]
                return image_goal

        # We have failed to find a suitable yaw. Return a NaN array.
        return np.array([np.nan])

    def _look_at_waypoint(self, waypoint: np.ndarray) -> np.ndarray:
        """
        (Currently unused)
        Returns the RGB image of the agent looking at a waypoint.
        """
        x_diff = waypoint[0] - self.agent_position[0]
        y_diff = waypoint[2] - self.agent_position[2]
        yaw = wrap_heading(-(np.arctan2(y_diff, x_diff) + np.pi / 2))
        look_at_rot = np.array([0, np.sin(yaw / 2), 0, np.cos(yaw / 2)])
        rgb = self._sim.get_observations_at(
            position=self.agent_position, rotation=look_at_rot
        )["rgb"]
        return rgb

    def _sample_exploration_start(self) -> bool:
        """
        Samples a starting pose for the agent to start exploring from

        Returns:
            bool: True if the path was successfully sampled, False otherwise.
        """

        def sample_position_from_same_floor():
            for _ in range(1000):
                sampled_position = self._sim.sample_navigable_point()
                self._sim.set_agent_state(
                    position=sampled_position, rotation=[0.0, 0.0, 0.0, 1.0]
                )

                path = get_path(
                    self._episode.start_position,
                    self._sim.get_agent_state().position,
                    self._sim,
                )
                if path is None:
                    continue

                if abs(self._sim.get_agent_state().position[1] - self._start_z) < 0.5:
                    return sampled_position
            raise RuntimeError("Failed to sample a valid point")

        success = False
        start = sample_position_from_same_floor()
        for attempt in range(self._max_exploration_attempts):
            path = get_path(self._episode.start_position, start, self._sim)
            z_values = [i[1] for i in path.points]
            if np.ptp(z_values) < 0.5:
                success = True
                break
            print(
                f"Floor traversal detected! Resampling... "
                f"{attempt + 1}/{self._max_exploration_attempts}"
            )

            start = sample_position_from_same_floor()

        return success

    def _decide_action(self, target: np.ndarray) -> np.ndarray:
        if self._is_exploring:
            max_steps_reached = (
                len(self._exploration_poses) >= self._max_exploration_steps
            )
            overshot = self._exploration_coverage > self._max_exploration_coverage
            if max_steps_reached or overshot:
                return ActionIDs.STOP
        return super()._decide_action(target)

    def _update_fog_of_war_mask(self):
        orig = self.fog_of_war_mask.copy()
        self._latest_fog = reveal_fog_of_war(
            top_down_map=self.top_down_map,
            current_fog_of_war_mask=np.zeros_like(self.fog_of_war_mask),
            current_point=self._get_agent_pixel_coords(),
            current_angle=self.agent_heading,
            fov=self._fov,
            max_line_len=self._visibility_dist_in_pixels,
        )
        # Update self.fog_of_war_mask with the new single_mask
        self.fog_of_war_mask[self._latest_fog == 1] = 1
        if self._is_exploring:
            self._exploration_fogs.append(self._latest_fog)
        updated = not np.array_equal(orig, self.fog_of_war_mask)

        if not self._is_exploring:
            if self._state == State.EXPLORE:
                # Start beelining if the minimum distance to the target is less than the
                # set threshold
                if self.check_explored_overlap():
                    self._state = State.BEELINE
                    self._beeline_target = self._valid_path.points[-1]

        return updated

    def update_frontiers(self):
        if not self._is_exploring:
            # Avoids filtering out frontiers in front of the goal
            super().update_frontiers()
            self._timestep_to_greedy_idx[self._step_count] = self.greedy_waypoint_idx
        else:
            # Avoids the actual goal for the episode affecting behavior
            BaseExplorer.update_frontiers(self)

    def _setup_pivot(self):
        if self._task_type == "objectnav":
            return ObjNavExplorer._setup_pivot(self)
        elif self._task_type == "imagenav":
            pass  # no need to pivot for ImageNav
        else:
            raise NotImplementedError

    def _pivot(self):
        if self._task_type == "objectnav":
            return ObjNavExplorer._pivot(self)
        elif self._task_type == "imagenav":
            return ActionIDs.STOP
        else:
            raise NotImplementedError

    def _reset_exploration(self) -> bool:
        """
        Resets the exploration process by sampling a new starting pose and updating the
        exploration goal point.

        Returns:
            bool: True if the exploration was successfully reset, False otherwise.
        """
        if not self._gt_traj:
            return False

        self._area_thresh = self._config.exploration_area_thresh
        self._visibility_dist = self._config.exploration_visibility_dist

        self.closest_frontier_waypoint = None
        self.frontier_waypoints = np.array([])
        self._exploration_poses = []
        self._exploration_imgs = []
        self._exploration_fogs = []
        self.fog_of_war_mask = np.zeros_like(self.top_down_map)
        self._agent_position = None
        self._agent_heading = None
        self._exploration_successful = False
        self._first_frontier = False
        self._minimize_time = True
        self._viz_imgs = []

        return self._sample_exploration_start()

    def _save_to_dataset(self) -> None:
        """
        Saves the frontier information to disk.
        """
        if "NUM_EXP_EPISODES" in os.environ:
            num_episodes = int(os.environ["NUM_EXP_EPISODES"])
            curr_num_episodes = len(
                glob.glob(f"{self._dataset_path}/{self._scene_id}/*")
            )
            msg = f"Finished {curr_num_episodes + 1} of {num_episodes} episodes"
            if curr_num_episodes >= num_episodes:
                print("Reached the desired number of episodes. Stopping...")
                import sys

                sys.exit(0)
        else:
            msg = ""

        if self._viz_imgs:
            output_path = (
                f"{os.environ['SAVE_VIZ']}/exp_{self._task_type}_{self._scene_id}_"
                f"{self._episode.episode_id}.mp4"
            )
            self._flush_visualization_images(output_path)

        # 'episode_dir' path should be {self._dataset_path}/{scene_id}/{episode_id}
        if osp.exists(self._episode_dir):
            # Episode already exists; skip saving
            return

        os.makedirs(self._episode_dir, exist_ok=True)

        frontiers = get_frontier_sets(self._gt_traj)
        if "all_frontiers" not in frontiers:
            # No frontier sets with enough frontiers were found
            return

        self._save_rgbs_to_video(
            [i.rgb for i in self._gt_traj], osp.join(self._episode_dir, "gt_traj.mp4")
        )
        self._save_frontier_fogs(self._episode_dir)
        if self._task_type == "imagenav":
            self._save_imagenav_goal(self._episode_dir)

        exploration_id = len(glob.glob(f"{self._episode_dir}/exploration_imgs_*"))
        exploration_imgs_dir = osp.join(
            self._episode_dir, f"exploration_imgs_{exploration_id}"
        )
        self._save_rgbs_to_video(
            self._exploration_imgs, osp.join(exploration_imgs_dir, "exploration.mp4")
        )
        self._save_exploration_fogs(exploration_imgs_dir)

        episode_json = osp.join(self._episode_dir, f"exploration_{exploration_id}.json")
        self._save_episode_json(frontiers, exploration_imgs_dir, episode_json)
        # self._save_coverage_visualization(exploration_imgs_dir)

        if msg:
            print(msg)

    def _save_coverage_visualization(self, exploration_imgs_dir: str) -> None:
        # Save visualization of the coverage
        h, w = self.fog_of_war_mask.shape
        coverage_img = np.zeros((h, w, 3), dtype=np.uint8)
        coverage_img[self.top_down_map == 1] = (255, 255, 255)  # White for free space
        coverage_img[self.fog_of_war_mask == 1] = (0, 0, 255)  # Blue for explored
        coverage_img[self._gt_fog_of_war_mask == 1] = (0, 255, 0)  # Green for GT
        coverage_img[  # Purple for overlap
            np.logical_and(self.fog_of_war_mask == 1, self._gt_fog_of_war_mask == 1)
        ] = (255, 0, 255)
        coverage_img = add_text_to_image(
            coverage_img, f"{self._exploration_coverage * 100 :.2f}%"
        )
        cv2.imwrite(f"{exploration_imgs_dir}/coverage.jpg", coverage_img)

    @staticmethod
    def _save_rgbs_to_video(rgbs: list[np.ndarray], output_path: str) -> None:
        parent_dir = osp.dirname(output_path)
        if not osp.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        bgr = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in rgbs]
        images_to_video(bgr, output_path)

    def _save_exploration_fogs(self, dir_path: str) -> None:
        self._save_fogs(self._exploration_fogs, dir_path, "exploration")

    def _save_frontier_fogs(self, dir_path: str) -> None:
        self._save_fogs(
            [i.single_fog_of_war for i in self._gt_traj],
            dir_path,
            "frontiers",
        )

    @staticmethod
    def _save_fogs(fogs, dir_path: str, prefix: str) -> None:
        fog_stack = np.stack(fogs)
        orig_shape = fog_stack.shape
        assert len(orig_shape) == 3
        shape_str = "_".join(str(i) for i in orig_shape)
        packed = np.packbits(fog_stack)
        filepath = osp.join(dir_path, f"{prefix}_{shape_str}.npy")
        np.save(filepath, packed)

    def _save_imagenav_goal(self, frontier_imgs_dir: str) -> None:
        filepath = osp.join(frontier_imgs_dir, "goal.jpg")
        bgr = cv2.cvtColor(self._imagenav_goal, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr)

    def _save_episode_json(
        self, frontiers: Dict, exploration_imgs_dir: str, episode_json: str
    ) -> None:
        """
        Saves the episode information to a JSON file.

        Args:
            frontiers (Dict): Dictionary representing the frontier sets.
            exploration_imgs_dir (str): The path to the directory to save the
                exploration
            episode_json (str): The path to the JSON file to save the episode
                information.
        """

        assert len(self._exploration_poses) == len(self._exploration_imgs), (
            f"{len(self._exploration_poses)=} " f"{len(self._exploration_imgs)=}"
        )

        # Save info to be able to map 3D points back to 2D
        lower_bound, upper_bound = self._sim.pathfinder.get_bounds()
        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / self.top_down_map.shape[0],
            abs(upper_bound[0] - lower_bound[0]) / self.top_down_map.shape[1],
        )
        json_data = {
            "episode_id": self._episode.episode_id,
            "scene_id": self._scene_id,
            "exploration_id": int(osp.basename(exploration_imgs_dir).split("_")[-1]),
            "exploration_poses": self._exploration_poses,
            "distance_to_goal": [s.distance_to_goal for s in self._gt_traj],
            "frontier_dtgs": [s.frontier_dtgs for s in self._gt_traj],
            "gt_poses": [s.pose for s in self._gt_traj],
            "lower_bound": [float(i) for i in lower_bound],
            "grid_size": [float(i) for i in grid_size],
            "start_z": self._start_z,
            **frontiers,
        }

        if self._task_type == "objectnav":
            json_data["object_category"] = self._episode.object_category

        with open(episode_json, "w") as f:
            print("Saving episode to:", episode_json)
            json.dump(json_data, f, indent=2)
            f.flush()

    def _flush_visualization_images(self, output_path: str) -> None:
        print(f"Writing visualization images to video at {output_path}...")
        self._viz_imgs = pad_images_to_max_dim(self._viz_imgs)
        parent_dir = osp.dirname(os.path.abspath(output_path))
        os.makedirs(parent_dir, exist_ok=True)
        images_to_video(self._viz_imgs, output_path, fps=5)
        self._viz_imgs = []

    @property
    def _exploration_coverage(self):
        return check_mask_overlap(
            self._gt_fog_of_war_mask,
            self.fog_of_war_mask,
        )

    @property
    def _is_unique_episode(self) -> bool:
        return not osp.exists(self._episode_dir)

    @property
    def _episode_dir(self) -> str:
        return str(
            osp.join(
                self._dataset_path,
                self._scene_id,
                f"episode_{self._episode.episode_id}",
            )
        )

    @default_on_exception(default_value=ActionIDs.STOP)
    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        if not self._is_exploring:
            super()._pre_step(episode)  # TargetExplorer._pre_step()
        else:
            BaseExplorer._pre_step(self, episode)

        if self._state == State.CANCEL or (
            self._unique_episodes_only and not self._is_unique_episode
        ):
            print(
                "STOP: One of these is true:\n"
                f"\t{self._state == State.CANCEL = }\n"
                f"\t{self._unique_episodes_only and not self._is_unique_episode = }"
            )
            task.is_stop_called = True
            return ActionIDs.STOP

        if not self._is_exploring:
            if self._stop_at_beelining and self._state == State.BEELINE:
                print("STOP: Beelining")
                action = ActionIDs.STOP
            else:
                if self._state != State.EXPLORE:
                    # These functions are only called when self._state == State.EXPLORE,
                    # but for this class, we want to call it every step
                    self.update_frontiers()
                    if self._should_update_closest_frontier:
                        i = self._get_closest_waypoint_idx()
                        self.closest_frontier_waypoint = self.frontier_waypoints[i]
                action = super().get_observation(task, episode, *args, **kwargs)
                if np.array_equal(action, ActionIDs.STOP):
                    print("STOP: Ground truth path completed.")
        else:
            # BaseExplorer already calls update_frontiers() and get_closest_waypoint()
            # at every step no matter what, so we don't need to call them here
            print(f"Exploring: {len(self._exploration_poses)}")
            action = BaseExplorer.get_observation(self, task, episode, *args, **kwargs)
            self._exploration_poses.append(self._curr_pose)

        stop_called = np.array_equal(action, ActionIDs.STOP)

        if not self._is_exploring and self._state != State.BEELINE:
            self._record_frontiers()
            print(
                f"GT path: {len(self._gt_traj)} "
                f"# frontiers: {len(self._gt_traj[self._step_count].all_frontiers)} "
            )
            if not stop_called and "SAVE_VIZ" in os.environ:
                self._viz_imgs.append(self._visualize_map())
        else:
            if len(self._exploration_poses) == 1:
                rgb = self._sim.get_observations_at()["rgb"]
            else:
                rgb = kwargs["observations"]["rgb"]
            self._exploration_imgs.append(rgb)

        # An episode is considered bad if the agent has timed out despite the episode
        # being feasible. However, since this sensor is always called before the map is
        # updated, we have to make sure that self._step_count is > 1
        if self._step_count == 0:
            self._bad_episode = False
            if self._task_type == "imagenav" and np.isnan(self._imagenav_goal).all():
                # Failed to find a suitable yaw for the goal image
                print("STOP: Failed to find a suitable yaw for the goal image.")
                task.is_stop_called = True
                return ActionIDs.STOP
        elif not self._is_exploring:  # Can stop checking once we start exploring
            if self._task_type == "objectnav":
                feasible = self._map_measure.get_metric()["is_feasible"]
                if feasible:
                    self._bad_episode = not stop_called
                else:
                    self._bad_episode = False
                    task.is_stop_called = True
                    print("STOP: Episode is infeasible; floor traversal required.")
                    return ActionIDs.STOP
            else:
                feasible = True

            if feasible and stop_called:
                # Ground truth path completed; move on to exploration phase
                if self._viz_imgs:
                    output_path = (
                        f"{os.environ['SAVE_VIZ']}/gt_{self._task_type}_"
                        f"{self._scene_id}_{self._episode.episode_id}.mp4"
                    )
                    self._flush_visualization_images(output_path)

                if (
                    not self._gt_traj
                    or max([len(i.all_frontiers) for i in self._gt_traj]) < 2
                ):
                    print("Only one frontier or less at each timestep!")
                    task.is_stop_called = True
                    return ActionIDs.STOP

                self._is_exploring = True
                self._gt_fog_of_war_mask = self.fog_of_war_mask.copy()
                print("Ground truth path completed! Resetting exploration.")

                # Reset the exploration
                success = self._reset_exploration()
                if "SAVE_VIZ" in os.environ:
                    self._viz_imgs.append(self._visualize_map())
                if not success:
                    # Could not find a valid exploration path
                    print("No valid exploration path found!")
                    task.is_stop_called = True
                    return ActionIDs.STOP
                else:
                    return self.get_observation(task, episode, *args, **kwargs)
        else:
            # Exploration is active. Check if exploration has successfully completed.

            if "SAVE_VIZ" in os.environ:
                self._viz_imgs.append(self._visualize_map())

            if np.array_equal(action, ActionIDs.STOP):
                # - The length of self._exploration_poses must be within the valid range
                # - The coverage of the exploration must be within the valid range
                self._exploration_successful = (
                    self._min_exploration_steps
                    <= len(self._exploration_poses)
                    <= self._max_exploration_steps + 1
                )

                if self._exploration_successful:
                    print("Exploration successful!")
                    # Save the exploration data to disk
                    self._save_to_dataset()
                else:
                    print("Exploration failed!")
                    success = self._reset_exploration()
                    if not success:
                        # Could not find a valid exploration path
                        task.is_stop_called = True
                        return ActionIDs.STOP
                    else:
                        return self.get_observation(task, episode, *args, **kwargs)
        if stop_called:
            task.is_stop_called = True

        self._step_count += 1

        return action

    def _visualize_map(self) -> np.ndarray:
        vis = super()._visualize_map()

        if not self._is_exploring:
            # Draw the shortest path to the closest goal in green
            sp = np.array(self._valid_path.points)
            self._draw_path(vis, sp, color=(0, 255, 0))
            for i, j in self._all_frontier_paths:
                i_p = np.array(i.points)
                j_p = np.array(j.points)
                self._draw_path(vis, i_p, color=(255, 0, 0))
                self._draw_path(vis, j_p, color=(0, 0, 255))

        valid_greedy_frontier_exists = (
            not self._is_exploring
            and len(self.frontier_waypoints) > 0
            and not np.isnan(self.frontier_waypoints).any()
        )
        if valid_greedy_frontier_exists:
            gt_idx = self.greedy_frontier_idx

            # Visualize its frontier segment in purple
            self._draw_path(
                vis, self._frontier_segments[gt_idx], color=(255, 0, 255), thickness=2
            )
            # Draw frontier midpoint
            greedy_px = self._vsf(self.frontier_waypoints[gt_idx][::-1])
            cv2.circle(vis, greedy_px, 4, (254, 254, 254), -1)
            cv2.circle(vis, greedy_px, 4, (0, 0, 0), 2)

            # Visualize how frontiers may have filtered other frontiers
            for bad_idx, good_idx in self._bad_idx_to_good_idx.items():
                bad_waypoint = self._vsf(self.frontier_waypoints[bad_idx][::-1])
                good_waypoint = self._vsf(self.frontier_waypoints[good_idx][::-1])
                cv2.circle(vis, bad_waypoint, 4, (200, 200, 200), 2)
                cv2.line(vis, bad_waypoint, good_waypoint, (0, 0, 255), 1)

        if not self._is_exploring:
            curr_state = self._gt_traj[self._step_count]

            # Visualize the highest scoring fow for each good frontier
            # Generate a mask representing gray and white pixels to color in
            gray_mask = np.all(
                vis == np.array([128, 128, 128], dtype=vis.dtype), axis=2
            )
            white_mask = np.all(
                vis == np.array([255, 255, 255], dtype=vis.dtype), axis=2
            )
            for t_step in curr_state.all_frontiers:
                if t_step == curr_state.correct_frontier:  # magenta
                    color = np.array([255, 216, 255], dtype=vis.dtype)
                else:  # orange
                    color = np.array([182, 238, 255], dtype=vis.dtype)

                fow = self.frontier_filter.get_fog_of_war(t_step)
                fow, _ = resize_image_maintain_ratio(
                    fow,
                    target_size=self._vis_map_height,
                    interpolation=cv2.INTER_NEAREST,
                )
                vis[(fow & (gray_mask | white_mask)).astype(bool)] = color
        else:
            curr_state = None

        # Remove unnecessary parts of the map (black)
        top_down_map, _ = resize_image_maintain_ratio(
            self.top_down_map,
            target_size=self._vis_map_height,
            interpolation=cv2.INTER_NEAREST,
        )
        smaller_contours = get_mask_except_nearest_contour(
            top_down_map, self._vsf(self._get_agent_pixel_coords())
        )
        vis[smaller_contours] = np.array([0, 0, 0], dtype=vis.dtype)
        not_black = ~np.all(vis == np.array([0, 0, 0], dtype=vis.dtype), axis=2)
        x, y, w, h = cv2.boundingRect(not_black.astype(np.uint8))
        y_min, y_max = max(0, y - 15), min(vis.shape[0], y + h + 15)
        x_min, x_max = max(0, x - 15), min(vis.shape[1], x + w + 15)
        vis = vis[y_min:y_max, x_min:x_max]

        rgb = self._sim.get_observations_at()["rgb"]
        other = (
            self._imagenav_goal
            if self._task_type == "imagenav"
            else np.ones_like(rgb) * 255
        )
        rgb = cv2.cvtColor(np.hstack([rgb, other]), cv2.COLOR_RGB2BGR)
        if curr_state is not None:
            # Orient it to 'portrait' mode
            vis = rotate_image_orientation(vis, portrait_mode=True)

            rgb, _ = resize_image_maintain_ratio(
                rgb,
                vis.shape[0] // 2,
                interpolation=cv2.INTER_AREA,
                use_shorter_dim=True,
            )
            # Stack the frontiers together in row major order
            f_imgs = []
            for t_step in curr_state.all_frontiers:
                f_img = cv2.cvtColor(self._gt_traj[t_step].rgb, cv2.COLOR_RGB2BGR)
                if t_step == curr_state.correct_frontier:
                    f_img = add_translucent_border(f_img, thickness=80)
                f_imgs.append(f_img)

            f_imgs_h, f_imgs_w = vis.shape[0] - rgb.shape[0], rgb.shape[1]
            if f_imgs:
                f_stacked = tile_images(np.array(f_imgs), max_width=4)
                f_imgs_final = place_image_centered(f_stacked, f_imgs_h, f_imgs_w)
            else:
                f_imgs_final = np.full((f_imgs_h, f_imgs_w, 3), 255, dtype=np.uint8)

            # Stack the images together
            vis = np.hstack([vis, np.vstack([rgb, f_imgs_final])])
        else:
            # Orient it to 'landscape' mode
            vis = rotate_image_orientation(vis, portrait_mode=False)
            vis, _ = resize_image_maintain_ratio(
                image=vis,
                target_size=rgb.shape[1],
                interpolation=cv2.INTER_LANCZOS4,
                use_shorter_dim=False,
            )
            vis = np.vstack([vis, rgb])

        # Add caption
        caption = f"Step: {self._step_count}"
        if self._task_type == "objectnav":
            caption += f" Object: {self._episode.object_category}"
        vis = add_text_to_image(vis, caption, top=True, above_padding=20)

        return vis


@dataclass(frozen=True)
class GTTrajectoryState:
    timestep_id: int
    pose: Tuple[float, float, float, float]
    rgb: np.ndarray
    all_frontiers: List[int]
    all_frontiers_unfiltered: List[int]
    all_frontiers_unscored: List[int]
    all_frontiers_unscored_unfiltered: List[int]
    correct_frontier: int
    correct_frontier_unscored: int
    single_fog_of_war: np.ndarray
    distance_to_goal: float
    frontier_dtgs: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def to_frontier_dict(self) -> Dict:
        return {
            "all_frontiers": [int(i) for i in self.all_frontiers],
            "all_frontiers_unfiltered": [int(i) for i in self.all_frontiers_unfiltered],
            "all_frontiers_unscored": [int(i) for i in self.all_frontiers_unscored],
            "all_frontiers_unscored_unfiltered": [
                int(i) for i in self.all_frontiers_unscored_unfiltered
            ],
            "correct_frontier": int(self.correct_frontier),
            "correct_frontier_unscored": int(self.correct_frontier_unscored),
        }


def get_frontier_sets(
    trajectory: List[GTTrajectoryState],
) -> Dict[str, Dict[str, List[int]]]:
    """
    Organizes trajectory states into sets of frontier options grouped by their
    configurations.

    This function processes a trajectory of robot states and groups timesteps with
    identical frontier selection options. It handles deduplication of frontier IDs when
    the robot revisits the same pose, ensuring that learning examples with identical
    decision points are properly grouped together.

    Args:
        trajectory: A list of GTTrajectoryState objects representing the robot's
                   exploration trajectory with frontier information at each timestep.

    Returns:
        A nested dictionary with the following structure:
        - First level keys are frontier types ("all_frontiers",
          "all_frontiers_unfiltered", etc.)
        - Second level keys are strings of format "correct_id|id1,id2,..." representing:
          * The correct frontier ID, followed by
          * A comma-separated list of all available frontier IDs at that configuration
        - Values are lists of timestep IDs that share the same frontier configuration
    """
    sets_to_timestep: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    set_keys = [
        "all_frontiers",
        "all_frontiers_unfiltered",
        "all_frontiers_unscored",
        "all_frontiers_unscored_unfiltered",
    ]
    pose_to_first_timestep: Dict[Tuple[float, float, float, float], int] = {}
    raw_timestep_to_first_timestep: Dict[int, int] = {}
    for state in trajectory:
        frontier_dict = state.to_frontier_dict()
        if state.pose not in pose_to_first_timestep:
            pose_to_first_timestep[state.pose] = state.timestep_id
        else:
            raw_timestep_to_first_timestep[state.timestep_id] = pose_to_first_timestep[
                state.pose
            ]
        for k in set_keys:
            correct_key = "_unscored" if "unscored" in k else ""
            correct_id = frontier_dict[f"correct_frontier{correct_key}"]
            correct_id = raw_timestep_to_first_timestep.get(correct_id, correct_id)
            deduped_choice_ids = sorted(
                [
                    str(raw_timestep_to_first_timestep.get(i, i))
                    for i in frontier_dict[k]
                ]
            )
            if len(deduped_choice_ids) < 2:
                continue
            set_string = f"{correct_id}|{','.join(deduped_choice_ids)}"
            sets_to_timestep[k][set_string].append(state.timestep_id)

    return sets_to_timestep


@dataclass
class ExplorationEpisodeGeneratorConfig(TargetExplorerSensorConfig):
    type: str = ExplorationEpisodeGenerator.__name__
    turn_angle: float = 30.0  # degrees
    forward_step_size: float = 0.5  # meters
    exploration_visibility_dist: float = 4.5  # meters
    visibility_dist: float = 2.15  # meters
    beeline_dist_thresh: float = 1.5  # meters
    success_distance: float = 0.5  # meters
    dataset_path: str = "data/exploration_episodes/"
    max_exploration_attempts: int = 100
    min_exploration_steps: int = 20
    max_exploration_steps: int = 2000
    min_exploration_coverage: float = 0.1
    max_exploration_coverage: float = 0.9
    exploration_area_thresh: float = 4.0
    unique_episodes_only: bool = True
    task_type: str = "objectnav"
    stop_at_beelining: bool = True
    no_flip_flopping: bool = True
    meters_per_pixel: float = 1.0 / 17.0


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.lab_sensors.exploration_episode_generator",
    group="habitat/task/lab_sensors",
    name="exploration_episode_generator",
    node=ExplorationEpisodeGeneratorConfig,
)


def check_mask_overlap(mask_1: np.ndarray, mask_2: np.ndarray) -> float:
    """
    Check if the overlap percentage between two binary masks is within a given range.

    Args:
        mask_1 (np.ndarray): A 2D numpy array containing binary values.
        mask_2 (np.ndarray): A 2D numpy array containing binary values, same shape as
            mask_1.

    Returns:
        float: The percentage of overlap between mask_1 and mask_2.

    Raises:
        AssertionError: If mask_1 and mask_2 have different shapes.
    """
    assert mask_1.shape == mask_2.shape, "Input masks must have the same shape."

    mask_1_set_pixels = np.sum(mask_1)
    overlapping_pixels = np.sum(np.logical_and(mask_1, mask_2))

    if mask_1_set_pixels > 0:
        overlap_percentage = overlapping_pixels / mask_1_set_pixels
    else:
        overlap_percentage = 0.0

    return overlap_percentage
