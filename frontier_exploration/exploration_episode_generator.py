from __future__ import annotations

import glob
import json
import os
import os.path as osp
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from habitat import EmbodiedTask, registry
from habitat.core.dataset import Episode
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.visualizations import maps
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import ActionIDs, BaseExplorer
from frontier_exploration.objnav_explorer import (
    GreedyObjNavExplorer,
    ObjNavExplorer,
    ObjNavExplorerSensorConfig,
    State,
)
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.frontier_filtering import FrontierInfo, filter_frontiers
from frontier_exploration.utils.general_utils import wrap_heading

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
class ExplorationEpisodeGenerator(ObjNavExplorer):
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

        self._gt_fog_of_war_mask: np.ndarray | None = None
        self._latest_fog: np.ndarray | None = None
        self._seen_frontier_sets: set = set()
        self._curr_frontier_set: set = set()
        self._frontier_sets: list[FrontierSet] = []
        self._gt_path_poses: list[list[float]] = []
        self._exploration_successful: bool = False
        self._start_z: float = -1.0
        self._max_frontiers: int = 0

        # This will just be used for debugging
        self._bad_episode: bool = False

    def _reset(self, episode: Episode) -> None:
        super()._reset(episode)

        self._visibility_dist = self._config.visibility_dist
        self._area_thresh = self._config.area_thresh

        self._is_exploring = False
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

    @property
    def _curr_pose(self) -> list[float]:
        quat = self._sim.get_agent_state().rotation
        yaw = 2 * np.arctan2(quat.y, quat.w)
        curr_pose = [*self._sim.get_agent_state().position, yaw]
        curr_pose = [float(f) for f in curr_pose]
        return curr_pose

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
        return GreedyObjNavExplorer._get_closest_waypoint(self)

    def _look_at_waypoint(self, waypoint: np.ndarray) -> np.ndarray:
        """
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
                if abs(self._sim.get_agent_state().position[1] - self._start_z) < 0.5:
                    return sampled_position
            raise RuntimeError("Failed to sample a valid point")

        success = False
        start = sample_position_from_same_floor()
        for attempt in range(self._max_exploration_attempts):
            # Start point must correspond to the same floor as the ground truth path
            sample_map = maps.get_topdown_map_from_sim(
                self._sim,
                map_resolution=self._map_resolution,
                draw_border=False,
            )
            if np.array_equal(sample_map, self.top_down_map):
                success = True
                rot = np.random.rand() * 2 * np.pi
                sampled_rotation = np.array([0, np.sin(rot / 2), 0, np.cos(rot / 2)])
                self._sim.set_agent_state(position=start, rotation=sampled_rotation)
                break
            print(
                f"Wrong floor! Resampling... "
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

    def _reset_exploration(self) -> bool:
        """
        Resets the exploration process by sampling a new starting pose and updating the
        exploration goal point.

        Returns:
            bool: True if the exploration was successfully reset, False otherwise.
        """
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

        return self._sample_exploration_start()

    def _save_to_dataset(self) -> None:
        """
        Saves the frontier information to disk.
        """
        # 'episode_dir' path should be {self._dataset_path}/{scene_id}/{episode_id}
        if not osp.exists(self._episode_dir):
            os.makedirs(self._episode_dir, exist_ok=True)
        else:
            return

        frontier_imgs_dir = osp.join(self._episode_dir, "frontier_imgs")
        self._save_frontier_images(frontier_imgs_dir)
        self._save_rgbs_to_video(
            self._gt_traj_imgs, osp.join(frontier_imgs_dir, "gt_traj.mp4")
        )
        self._save_frontier_fogs(frontier_imgs_dir)

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
        if "NUM_EXP_EPISODES" in os.environ:
            num_episodes = int(os.environ["NUM_EXP_EPISODES"])
            curr_num_episodes = len(
                glob.glob(f"{self._dataset_path}/{self._scene_id}/*")
            )
            print(f"Finished {curr_num_episodes} of {num_episodes} episodes")
            if curr_num_episodes >= num_episodes:
                print("Reached the desired number of episodes. Stopping...")
                quit()

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
            "object_category": self._episode.object_category,
            "gt_path_poses": self._gt_path_poses,
            "frontiers": frontiers,
            "timestep_to_frontiers": {
                fs.time_step: fs.to_dict() for fs in self._frontier_sets
            },
            "exploration_poses": self._exploration_poses,
            "exploration_imgs": sorted(
                [
                    osp.join(exploration_imgs_dir, f)
                    for f in glob.glob(f"{exploration_imgs_dir}/*.jpg")
                ]
            ),
        }
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

        if self._unique_episodes_only and not self._is_unique_episode:
            task.is_stop_called = True
            return ActionIDs.STOP

        if not self._is_exploring:
            if self._state != State.EXPLORE:
                # These functions are only called when self._state == State.EXPLORE, but
                # for this class, we want to call it every step
                self._update_frontiers()
                self.closest_frontier_waypoint = self._get_closest_waypoint()
            print(
                f"GT path: {len(self._gt_path_poses)} "
                f"# frontiers: {len(self._curr_frontier_set)} "
            )
            action = super().get_observation(task, episode, *args, **kwargs)
        else:
            # BaseExplorer already calls _update_frontiers() and get_closest_waypoint()
            # at every step no matter what, so we don't need to call them here
            print(f"Exploring: {len(self._exploration_poses)}")
            action = BaseExplorer.get_observation(self, task, episode, *args, **kwargs)

        stop_called = np.array_equal(action, ActionIDs.STOP)

        self._record_curr_pose()

        if not self._is_exploring:
            self._record_frontiers()
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
        elif not self._is_exploring:  # Can stop checking once we start exploring
            feasible = self._map_measure.get_metric()["is_feasible"]
            if feasible:
                self._bad_episode = not stop_called
            else:
                self._bad_episode = False  # infeasible, but not feasible + unsuccessful
                task.is_stop_called = True
                return ActionIDs.STOP

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


def images_to_video(image_list, output_path, fps=5):
    # Get the height and width from the first image
    height, width = image_list[0].shape[:2]
    print(f"Creating video with dimensions {width}x{height} at {fps} FPS")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image in image_list:
        out.write(image)

    # Release the VideoWriter
    out.release()

    print(f"Video saved to {output_path}")


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
class ExplorationEpisodeGeneratorConfig(ObjNavExplorerSensorConfig):
    type: str = ExplorationEpisodeGenerator.__name__
    turn_angle: float = 30.0  # degrees
    forward_step_size: float = 0.5  # meters
    beeline_dist_thresh: float = 2  # meters
    success_distance: float = 0.1  # meters
    dataset_path: str = "data/exploration_episodes/"
    max_exploration_attempts: int = 10
    min_exploration_steps: int = 20
    max_exploration_steps: int = 100
    min_exploration_coverage: float = 0.1
    max_exploration_coverage: float = 0.9
    exploration_visibility_dist: float = 5.5
    exploration_area_thresh: float = 4.0
    unique_episodes_only: bool = True


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


def add_text_to_image(image, text):
    # Get image dimensions
    height, width = image.shape[:2]

    font_size = 3

    # Set font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, 2)

    # Calculate white space height (120% of text height)
    white_space_height = int(text_height * 1.2)

    # Create new image with white space
    new_height = height + white_space_height
    new_image = np.zeros((new_height, width, 3), dtype=np.uint8)
    new_image[:white_space_height] = [255, 255, 255]  # White space
    new_image[white_space_height:] = image

    # Calculate text position
    text_x = (width - text_width) // 2
    text_y = int(white_space_height * 0.5 + text_height * 0.5)

    # Add text to image
    cv2.putText(
        new_image, text, (text_x, text_y), font, font_size, (0, 0, 0), 3, cv2.LINE_AA
    )

    return new_image
