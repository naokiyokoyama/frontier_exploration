from __future__ import annotations

import glob
import json
import os
import os.path as osp
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import quaternion as qt
from habitat import EmbodiedTask, registry
from habitat.core.dataset import Episode
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import ActionIDs, BaseExplorer, get_polar_angle
from frontier_exploration.objnav_explorer import ObjNavExplorer
from frontier_exploration.target_explorer import (
    GreedyExplorerMixin,
    State,
    TargetExplorer,
    TargetExplorerSensorConfig,
)
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.frontier_filtering import FrontierInfo, filter_frontiers
from frontier_exploration.utils.general_utils import images_to_video, wrap_heading
from frontier_exploration.utils.path_utils import get_path
from frontier_exploration.utils.viz import (
    add_text_to_image,
    add_translucent_green_border,
)

EXPLORATION_THRESHOLD = 0.1


def default_on_exception(default_value):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Exception occurred: {e}")
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

        # Fields for storing data that will be recorded into the dataset
        self._frontier_pose_to_id: dict = {}
        self._frontier_id_to_img: dict = {}
        self._exploration_poses: list[list[float]] = []
        self._exploration_imgs: list[np.ndarray] = []
        self._exploration_fogs: list[np.ndarray] = []
        self._gt_frontiers: list[FrontierInfo] = []  # all frontiers seen in the episode
        self._gt_traj_imgs: list[np.ndarray] = []
        self._seen_frontiers: set[tuple[int, int]] = set()

        self._gt_fog_of_war_mask: np.ndarray = np.empty((1, 1))  # 2D array
        self._latest_fog: np.ndarray = np.empty((1, 1))  # 2D array
        self._seen_frontier_sets: set = set()
        self._curr_frontier_set: set = set()
        self._frontier_sets: list[FrontierSet] = []
        self._gt_path_poses: list[list[float]] = []
        self._exploration_successful: bool = False
        self._start_z: float = -1.0
        self._max_frontiers: int = 0

        # For ImageNav
        self._imagenav_goal: np.ndarray = np.empty((1, 1))  # 2D array
        self._total_fov_pixels: int = 0

        # This will just be used for debugging
        self._bad_episode: bool = False
        self._viz_imgs: list[np.ndarray] = []

    def _reset(self, episode: Episode) -> None:
        super()._reset(episode)

        self._visibility_dist = self._config.visibility_dist
        self._area_thresh = self._config.area_thresh

        self._is_exploring = False
        self._minimize_time = False
        self._frontier_pose_to_id = {}
        self._frontier_id_to_img = {}

        self._gt_fog_of_war_mask = None
        self._latest_fog = None
        self._seen_frontier_sets = set()
        self._curr_frontier_set = set()
        self._exploration_poses = []
        self._exploration_imgs = []
        self._frontier_sets = []
        self._gt_path_poses = []
        self._exploration_successful = False
        self._gt_frontiers = []
        self._seen_frontiers = set()
        self._gt_traj_imgs = []
        self._max_frontiers = 0
        self._coverage_masks = []

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
            print(f"Episode {episode.episode_id} failed!!")
            filename = f"{episode.episode_id}_{self._scene_id}.txt"
            with open(filename, "w") as f:
                f.write("")
        elif self._step_count > 0:
            print(f"Episode {episode.episode_id} succeeded!!")

        self._bad_episode = False
        self._start_z = self._sim.get_agent_state().position[1]

    def _record_curr_pose(self) -> None:
        """
        Records the current pose of the agent.
        """
        # Record the current pose
        curr_pose = self._curr_pose
        if self._is_exploring:
            self._exploration_poses.append(curr_pose)
        else:
            self._gt_path_poses.append(curr_pose)

    def _record_frontiers(self) -> None:
        """
        Updates self._timestep_to_frontiers and self._frontier_pose_to_id with any new
        frontier information. Because the amount of new frontier images for one timestep
        can only be 0 or 1, the frontier id is simply set to the timestep.
        """
        if len(self.frontier_waypoints) == 0 or self._state == State.BEELINE:
            return  # No frontiers to record
        rgb = self._sim.get_observations_at()["rgb"]
        self._gt_traj_imgs.append(rgb)
        for f_position in self.frontier_waypoints:
            # For each frontier, convert its position to a tuple to make it hashable
            f_position_tuple: tuple[int, int] = tuple(f_position)  # noqa
            seen_pos_tuples = set(f.position_tuple for f in self._gt_frontiers)
            if f_position_tuple not in seen_pos_tuples:
                # Encountered a new frontier
                frontier_id = len(self._gt_frontiers)
                self._gt_frontiers.append(
                    FrontierInfo(
                        agent_pose=self._curr_pose,
                        camera_position_px=self._get_agent_pixel_coords(),
                        frontier_position=self._pixel_to_map_coors(f_position),
                        frontier_position_px=f_position_tuple,
                        single_fog_of_war=self._latest_fog,
                        rgb_img=rgb,
                    )
                )
                self._frontier_pose_to_id[f_position_tuple] = frontier_id

        # Update the current frontier set by cross-referencing self.frontier_waypoints
        # against self._frontier_pose_to_id
        active_ids = [
            self._frontier_pose_to_id[tuple(pose)] for pose in self.frontier_waypoints
        ]
        gt_idx = active_ids.index(
            self._frontier_pose_to_id[tuple(self._correct_frontier_waypoint)]
        )
        frontier_infos = [self._gt_frontiers[i] for i in active_ids]
        boundary_contour = cv2.findContours(
            self.fog_of_war_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        if len(boundary_contour) == 0:
            return  # No boundary contour found
        else:
            boundary_contour = boundary_contour[0]
        inds_to_keep = filter_frontiers(frontier_infos, boundary_contour, gt_idx)
        self._curr_frontier_set = frozenset(active_ids[i] for i in inds_to_keep)
        pos_set = frozenset(frontier_infos[i].agent_pose_tuple for i in inds_to_keep)
        self._max_frontiers = max(self._max_frontiers, len(self._curr_frontier_set))
        if pos_set not in self._seen_frontier_sets:
            # New frontier set found
            self._seen_frontier_sets.add(pos_set)
            self._frontier_sets.append(
                FrontierSet(
                    frontier_ids=list(self._curr_frontier_set),
                    best_id=self._frontier_pose_to_id[
                        tuple(self._correct_frontier_waypoint)
                    ],
                    time_step=self._step_count - 1,
                )
            )

    @property
    def _correct_frontier_waypoint(self) -> np.ndarray:
        return GreedyExplorerMixin._get_closest_waypoint(self)

    def _generate_imagenav_goal(self, episode: Episode) -> np.ndarray:
        """
        Generates an image taken at the goal point for ImageNav tasks. The yaw of the
        agent when the image is taken is selected such that at least 50% of the agent's
        FOV is unobstructed. This is to avoid image goals that are too close to walls.

        Args:
            episode (Episode): The current episode.

        Returns:
            np.ndarray: The RGB image of the goal.
        """
        if self._total_fov_pixels == 0:
            height, width = self.top_down_map.shape
            single_fov = reveal_fog_of_war(
                np.ones_like(self.top_down_map),  # fully navigable map
                np.zeros_like(self.fog_of_war_mask),  # no explored areas
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
                self.top_down_map,
                np.zeros_like(self.fog_of_war_mask),  # no explored areas
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
            self.top_down_map,
            np.zeros_like(self.fog_of_war_mask),
            self._get_agent_pixel_coords(),
            self.agent_heading,
            fov=self._fov,
            max_line_len=self._visibility_dist_in_pixels,
        )
        # Update self.fog_of_war_mask with the new single_mask
        self.fog_of_war_mask[self._latest_fog == 1] = 1
        if self._is_exploring:
            self._exploration_fogs.append(self._latest_fog)
        updated = not np.array_equal(orig, self.fog_of_war_mask)

        if not self._is_exploring:
            min_dist = self._get_min_dist()
            if self._state == State.EXPLORE:
                # Start beelining if the minimum distance to the target is less than the
                # set threshold
                if min_dist < self._beeline_dist_thresh:
                    self._state = State.BEELINE
                    self._beeline_target = self._episode._shortest_path_cache.points[-1]

        return updated

    def _update_frontiers(self):
        if not self._is_exploring:
            # Avoids filtering out frontiers in front of the goal
            super()._update_frontiers()
        else:
            # Avoids the actual goal for the episode affecting behavior
            BaseExplorer._update_frontiers(self)

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
        if self._viz_imgs:
            output_path = (
                f"{self._task_type}_{self._scene_id}_{self._episode.episode_id}.mp4"
            )
            print(f"Writing visualization images to video at {output_path}...")
            self._viz_imgs = pad_images_to_max_height(self._viz_imgs)
            images_to_video(self._viz_imgs, output_path, fps=5)

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

        # 'episode_dir' path should be {self._dataset_path}/{scene_id}/{episode_id}
        if osp.exists(self._episode_dir):
            return

        os.makedirs(self._episode_dir, exist_ok=True)

        frontier_imgs_dir = osp.join(self._episode_dir, "frontier_imgs")
        self._save_frontier_images(frontier_imgs_dir)
        self._save_rgbs_to_video(
            self._gt_traj_imgs, osp.join(frontier_imgs_dir, "gt_traj.mp4")
        )
        self._save_frontier_fogs(frontier_imgs_dir)
        if self._task_type == "imagenav":
            self._save_imagenav_goal(frontier_imgs_dir)

        exploration_id = len(glob.glob(f"{self._episode_dir}/exploration_imgs_*"))
        exploration_imgs_dir = osp.join(
            self._episode_dir, f"exploration_imgs_{exploration_id}"
        )
        self._save_rgbs_to_video(
            self._exploration_imgs, osp.join(exploration_imgs_dir, "exploration.mp4")
        )
        self._save_exploration_fogs(exploration_imgs_dir)

        episode_json = osp.join(self._episode_dir, f"exploration_{exploration_id}.json")
        self._save_episode_json(self._episode_dir, exploration_imgs_dir, episode_json)
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

    def _save_frontier_images(self, frontier_imgs_dir: str) -> None:
        """
        Saves the frontier images to disk.

        Args:
            frontier_imgs_dir (str): The path to the directory to save the frontier
                images.
        """
        if not osp.exists(frontier_imgs_dir):
            os.makedirs(frontier_imgs_dir, exist_ok=True)
        for frontier_id, f_info in enumerate(self._gt_frontiers):
            # Only save the frontier if it is a member of at least one FrontierSet
            # that has at least 2 frontiers
            necessary = False
            for frontier_set in self._frontier_sets:
                if frontier_id in frontier_set.frontier_ids and len(frontier_set) > 1:
                    necessary = True
                    break

            if not necessary:
                continue

            img_filename = osp.join(
                frontier_imgs_dir, f"frontier_{frontier_id:04d}.jpg"
            )
            img_bgr = cv2.cvtColor(f_info.rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_filename, img_bgr)

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
            [f.single_fog_of_war for f in self._gt_frontiers], dir_path, "frontiers"
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
        self, frontier_imgs_dir: str, exploration_imgs_dir: str, episode_json: str
    ) -> None:
        """
        Saves the episode information to a JSON file.

        Args:
            frontier_imgs_dir (str): The path to the directory to save the frontier
                images.
            exploration_imgs_dir (str): The path to the directory to save the
                exploration
            episode_json (str): The path to the JSON file to save the episode
                information.
        """
        frontiers = {}
        for f_id, f_info in enumerate(self._gt_frontiers):
            frontiers.update(f_info.to_dict(f_id, frontier_imgs_dir))

        assert len(self._exploration_poses) == len(self._exploration_imgs), (
            f"{len(self._exploration_poses)=} " f"{len(self._exploration_imgs)=}"
        )
        json_data = {
            "episode_id": self._episode.episode_id,
            "scene_id": self._scene_id,
            "exploration_id": int(osp.basename(exploration_imgs_dir).split("_")[-1]),
            "gt_path_poses": self._gt_path_poses,
            "frontiers": frontiers,
            "timestep_to_frontiers": {
                fs.time_step: fs.to_dict() for fs in self._frontier_sets
            },
            "exploration_poses": self._exploration_poses,
        }

        if self._task_type == "objectnav":
            json_data["object_category"] = self._episode.object_category

        with open(episode_json, "w") as f:
            print("Saving episode to:", episode_json)
            json.dump(json_data, f)

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
            super()._pre_step(episode)
        else:
            BaseExplorer._pre_step(self, episode)

        if self._state == State.CANCEL or (
            self._unique_episodes_only and not self._is_unique_episode
        ):
            print(
                "STOP: One of these is true: "
                f"{self._state == State.CANCEL = }, "
                f"{self._unique_episodes_only and not self._is_unique_episode = }"
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
                    self._update_frontiers()
                    self.closest_frontier_waypoint = self._get_closest_waypoint()
                print(
                    f"GT path: {len(self._gt_path_poses)} "
                    f"# frontiers: {len(self._curr_frontier_set)} "
                )
                action = super().get_observation(task, episode, *args, **kwargs)
                if np.array_equal(action, ActionIDs.STOP):
                    print("STOP: Ground truth path completed.")
        else:
            # BaseExplorer already calls _update_frontiers() and get_closest_waypoint()
            # at every step no matter what, so we don't need to call them here
            print(f"Exploring: {len(self._exploration_poses)}")
            action = BaseExplorer.get_observation(self, task, episode, *args, **kwargs)

        stop_called = np.array_equal(action, ActionIDs.STOP)

        self._record_curr_pose()

        if not self._is_exploring:
            self._record_frontiers()
            # self._visualize_map()
        else:
            if len(self._exploration_poses) == 1:
                rgb = self._sim.get_observations_at()["rgb"]
            else:
                rgb = kwargs["observations"]["rgb"]
            self._exploration_imgs.append(rgb)

        # An episode is considered bad if the agent has timed out despite the episode
        # being feasible. However, since this sensor is always called before the map is
        # updated, we have to make sure that self._step_count is > 1
        if self._step_count == 1:
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

                if self._max_frontiers < 2:
                    print("Not enough frontiers!")
                    task.is_stop_called = True
                    return ActionIDs.STOP

                self._is_exploring = True
                self._gt_fog_of_war_mask = self.fog_of_war_mask.copy()
                print("Ground truth path completed! Resetting exploration.")

                # Reset the exploration
                success = self._reset_exploration()
                if not success:
                    # Could not find a valid exploration path
                    print("No valid exploration path found!")
                    task.is_stop_called = True
                    return ActionIDs.STOP
                else:
                    return self.get_observation(task, episode, *args, **kwargs)
        else:
            # Exploration is active. Check if exploration has successfully completed.
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

        return action

    def _visualize_map(self, exploration: bool = False):
        if self._step_count <= 1:
            return

        top_down_map_vis = self.top_down_map.astype(np.uint8) * 255
        top_down_map_vis[self.fog_of_war_mask == 1] = 128
        top_down_map_vis = cv2.cvtColor(top_down_map_vis, cv2.COLOR_GRAY2BGR)

        # Draw the current agent position as a filled green circle of size 4
        agent_px = self._get_agent_pixel_coords()[::-1]
        cv2.circle(top_down_map_vis, tuple(agent_px), 4, (0, 255, 0), -1)

        if not exploration:
            # Draw the goal point as a filled red circle of size 4
            goal_px = self._map_coors_to_pixel(self._closest_goal)[::-1]
            cv2.circle(top_down_map_vis, tuple(goal_px), 4, (0, 0, 255), -1)

            # Draw the beeline radius circle (not filled) in blue, convert meters to pixels
            beeline_radius = self._convert_meters_to_pixel(self._beeline_dist_thresh)
            cv2.circle(top_down_map_vis, tuple(goal_px), beeline_radius, (255, 0, 0), 1)

            # Draw the success radius circle in green
            success_radius = self._convert_meters_to_pixel(
                self._config.success_distance
            )
            cv2.circle(top_down_map_vis, tuple(goal_px), success_radius, (0, 255, 0), 1)

        # For each frontier waypoint, draw an unfilled circle in orange, or blue if
        # it's the chosen waypoint.
        for waypoint in self.frontier_waypoints:
            color = (0, 165, 255)  # orange
            if tuple(waypoint) == tuple(self._correct_frontier_waypoint):
                color = (255, 0, 0)
            cv2.circle(top_down_map_vis, waypoint[::-1].astype(np.int32), 3, color, -1)

        rgb = self._sim.get_observations_at()["rgb"]

        if self._task_type == "imagenav":
            rgb = np.hstack([rgb, self._imagenav_goal])
        else:
            goal = np.ones_like(rgb) * 255
            goal = add_text_to_image(goal, self._episode.object_category)
            goal = goal[: rgb.shape[0], :]
            rgb = np.hstack([rgb, goal])

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        map_height, map_width = top_down_map_vis.shape[:2]
        scaled_width = int(rgb.shape[1] * map_height / rgb.shape[0])
        rgb_resized = cv2.resize(rgb, (scaled_width, map_height))
        img = np.hstack([rgb_resized, top_down_map_vis])
        img = add_text_to_image(
            img,
            f"# frontiers: {len(self._curr_frontier_set)}  "
            f"Step: {self._step_count}",
        )

        # Stack the frontiers towards the bottom
        curr_frontier_ids = list(self._curr_frontier_set)
        if curr_frontier_ids:
            curr_frontier_imgs = []
            gt_i = self._frontier_pose_to_id.get(tuple(self._correct_frontier_waypoint))
            for i in curr_frontier_ids:
                f_img = cv2.cvtColor(self._gt_frontiers[i].rgb_img, cv2.COLOR_RGB2BGR)
                if not exploration and i == gt_i:
                    f_img = add_translucent_green_border(f_img)
                curr_frontier_imgs.append(f_img)

            num_frontier_width = minimize_difference(map_width, scaled_width) + 2

            curr_frontier_imgs = stack_images(curr_frontier_imgs, num_frontier_width)
            curr_frontier_imgs = resize_image(curr_frontier_imgs, img.shape[1])

            img = np.vstack([img, curr_frontier_imgs])

        self._viz_imgs.append(img)


def pad_images_to_max_height(images):
    # Assert all images have the same width
    widths = [img.shape[1] for img in images]
    assert len(set(widths)) == 1, "All images must have the same width"

    # Find the maximum height
    max_height = max(img.shape[0] for img in images)

    # Pad images to the maximum height
    padded_images = []
    for img in images:
        height_diff = max_height - img.shape[0]
        if height_diff > 0:
            # Create a white padding
            padding = np.full((height_diff, img.shape[1], 3), 255, dtype=np.uint8)
            # Concatenate the original image with the padding
            padded_img = np.vstack((img, padding))
        else:
            padded_img = img
        padded_images.append(padded_img)

    return padded_images


def resize_image(image, new_width):
    # Get the original dimensions
    height, width = image.shape[:2]

    # Calculate the ratio of the new width to the old width
    ratio = new_width / float(width)

    # Calculate the new height to maintain the aspect ratio
    new_height = int(height * ratio)

    # Resize the image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    return resized_image


def minimize_difference(x, y):
    if y == 0:
        return max(1, x)

    closest = max(1, round(x / y))
    return closest


def stack_images(images, N):
    if not images or N <= 0:
        raise ValueError(
            "Invalid input: images list must not be empty and N must be positive"
        )

    height, width = images[0].shape[:2]
    rows = (len(images) - 1) // N + 1
    result_height = height * rows
    result_width = width * N

    # Create a white canvas
    result = np.ones((result_height, result_width, 3), dtype=np.uint8) * 255

    for i, img in enumerate(images):
        row = i // N
        col = i % N
        y_start = row * height
        x_start = col * width
        result[y_start : y_start + height, x_start : x_start + width] = img

    return result


class FrontierSet:
    def __init__(self, frontier_ids: list[int], best_id: int, time_step: int):
        assert best_id in frontier_ids
        self._best_id = best_id
        self.frontier_ids = frontier_ids
        self.time_step = time_step

    def __len__(self):
        return len(self.frontier_ids)

    def to_dict(self):
        return {
            "frontier_ids": self.frontier_ids,
            "best_id": self._best_id,
        }


@dataclass
class ExplorationEpisodeGeneratorConfig(TargetExplorerSensorConfig):
    type: str = ExplorationEpisodeGenerator.__name__
    turn_angle: float = 30.0  # degrees
    forward_step_size: float = 0.5  # meters
    exploration_visibility_dist: float = 4.5  # meters
    visibility_dist: float = 2.15  # meters
    beeline_dist_thresh: float = 2.25  # meters; > visibility_dist required
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
