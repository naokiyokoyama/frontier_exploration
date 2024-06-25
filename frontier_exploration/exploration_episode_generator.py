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
from habitat.datasets.pointnav.pointnav_generator import _ratio_sample_rate
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.visualizations import maps
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import ActionIDs, BaseExplorer
from frontier_exploration.objnav_explorer import (
    GreedyObjNavExplorer,
    ObjNavExplorerSensorConfig,
    State,
)
from frontier_exploration.utils.path_utils import get_path


@registry.register_sensor
class ExplorationEpisodeGenerator(GreedyObjNavExplorer):
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
        self._max_exploration_attempts = config.max_exploration_attempts
        self._min_exploration_steps = config.min_exploration_steps
        self._max_exploration_steps = config.max_exploration_steps
        self._min_exploration_coverage = config.min_exploration_coverage
        self._max_exploration_coverage = config.max_exploration_coverage
        self._map_measure = task.measurements.measures["top_down_map"]

        self._is_exploring: bool = False

        # Fields for storing data that will be recorded into the dataset
        self._timestep_to_frontiers: dict = {}
        self._frontier_pose_to_id: dict = {}
        self._frontier_id_to_img: dict = {}
        self._exploration_poses: list[list[float]] = []
        self._exploration_imgs: list[np.ndarray] = []

        self._gt_fog_of_war_mask: np.ndarray | None = None
        self._seen_frontier_sets: set = set()
        self._curr_frontier_set: set = set()
        self._frontier_sets: list[FrontierSet] = []
        self._gt_path_poses: list[list[float]] = []
        self._exploration_coverage: float = -1.0
        self._exploration_successful: bool = False
        self._start_z: float = -1.0

        # This will just be used for debugging
        self._bad_episode: bool = False

    def _reset(self, episode: Episode) -> None:
        super()._reset(episode)

        self._is_exploring = False
        self._timestep_to_frontiers = {}
        self._frontier_pose_to_id = {}
        self._frontier_id_to_img = {}

        self._gt_fog_of_war_mask = None
        self._seen_frontier_sets = set()
        self._curr_frontier_set = set()
        self._exploration_poses = []
        self._exploration_imgs = []
        self._frontier_sets = []
        self._gt_path_poses = []
        self._exploration_coverage = -1.0
        self._exploration_successful = False

        # If the last episode failed, then we need to record the episode's id and its
        # scene id for further debugging
        assert hasattr(episode, "episode_id") and hasattr(
            episode, "scene_id"
        ), "Episode must have episode_id and scene_id attributes"
        if self._bad_episode:
            # Create a blank file in the cwd with the episode id and scene id
            print(f"Episode {episode.episode_id} failed!!")
            scene_id = extract_scene_id(episode)
            filename = f"{episode.episode_id}_{scene_id}.txt"
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
        quat = self._sim.get_agent_state().rotation
        yaw = 2 * np.arctan2(quat.y, quat.w)
        curr_pose = [*self._sim.get_agent_state().position, yaw]
        curr_pose = [float(f) for f in curr_pose]
        if self._is_exploring:
            self._exploration_poses.append(curr_pose)
        else:
            self._gt_path_poses.append(curr_pose)

    def _record_frontiers(self, rgb: np.ndarray) -> None:
        """
        Updates self._timestep_to_frontiers and self._frontier_pose_to_id with any new
        frontier information. Because the amount of new frontier images for one timestep
        can only be 0 or 1, the frontier id is simply set to the timestep.
        """
        if len(self.frontier_waypoints) == 0:
            return
        frontier_id = self._step_count - 1
        for pose in self.frontier_waypoints:
            # For each frontier, convert its pose to a tuple to make it hashable
            pose = tuple(pose)
            if pose not in self._frontier_pose_to_id:
                # New frontier found
                self._frontier_pose_to_id[pose] = frontier_id
                # Assign image to current frontier_id (timestep) if not already done
                if frontier_id not in self._timestep_to_frontiers:
                    self._frontier_id_to_img[frontier_id] = rgb

        # Update the current frontier set by cross-referencing self.frontier_waypoints
        # against self._frontier_pose_to_id
        self._curr_frontier_set = set(
            self._frontier_pose_to_id[tuple(pose)] for pose in self.frontier_waypoints
        )
        if self._curr_frontier_set not in self._seen_frontier_sets:
            # New frontier set found
            best_id = self._frontier_pose_to_id[tuple(self.closest_frontier_waypoint)]
            self._frontier_sets.append(
                FrontierSet(
                    frontier_ids=list(self._curr_frontier_set),
                    best_id=best_id,
                    time_step=self._step_count - 1,
                )
            )
            self._seen_frontier_sets.add(tuple(self._curr_frontier_set))

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
        start, end = np.zeros(3), np.zeros(3)
        for _ in range(self._max_exploration_attempts):
            start = sample_position_from_same_floor()
            end = sample_position_from_same_floor()

            # Need to make sure that:
            # - a navigable path exists between the start and end points
            # - the start and end points are not too close to each other
            # - the ratio between the Euclidean distance and the shortest path distance
            #   is within a certain range
            path = get_path(start, end, self._sim)
            if path is None:
                continue  # No path found
            path_length = path.geodesic_distance
            if not 1 <= path_length <= 30:
                continue  # Path too short or too long
            euclid_dist = np.power(
                np.power(np.array(start) - np.array(end), 2).sum(0), 0.5
            )
            distances_ratio = path_length / euclid_dist
            if distances_ratio < 1.1 and (
                np.random.rand() > _ratio_sample_rate(distances_ratio, 1.1)
            ):
                continue  # Path too straight

            # Sampled trajectory has to overlap with the ground truth trajectory
            sampled_trajectory = self._get_trajectory_mask(path.points)
            if (
                check_mask_overlap(self._gt_fog_of_war_mask, sampled_trajectory)
                < self._min_exploration_coverage
            ):
                continue  # not enough overlap

            # Start point must correspond to the same floor as the ground truth path
            sample_map = maps.get_topdown_map_from_sim(
                self._sim,
                map_resolution=self._map_resolution,
                draw_border=False,
            )
            if not np.array_equal(sample_map, self.top_down_map):
                continue

            success = True
            break

        self._beeline_target = end
        self._state = State.BEELINE
        rot = np.random.rand() * 2 * np.pi
        sampled_rotation = np.array([0, np.sin(rot / 2), 0, np.cos(rot / 2)])
        self._sim.set_agent_state(position=start, rotation=sampled_rotation)

        return success

    def _decide_action(self, target: np.ndarray) -> np.ndarray:
        if self._is_exploring:
            if (
                self._get_min_dist() < self._success_distance
                or len(self._exploration_poses) >= self._max_exploration_steps
            ):
                return ActionIDs.STOP
        return super()._decide_action(target)

    def _update_fog_of_war_mask(self):
        updated = BaseExplorer._update_fog_of_war_mask(self)
        if not self._is_exploring:
            min_dist = self._get_min_dist()
            if self._state == State.EXPLORE:
                # Start beelining if the minimum distance to the target is less than the
                # set threshold
                if min_dist < self._beeline_dist_thresh:
                    self._state = State.BEELINE
                    self._beeline_target = self._episode._shortest_path_cache.points[-1]

        return updated

    def _get_min_dist(self):
        """Returns the minimum distance to the target"""
        if self._is_exploring:
            return self._sim.geodesic_distance(
                self._sim.get_agent_state().position, [self._beeline_target]
            )
        return super()._get_min_dist()

    def _reset_exploration(self) -> np.ndarray:
        """
        Resets the exploration process by sampling a new starting pose and updating the
        exploration goal point.

        Returns:
            np.ndarray: The action to take to start the exploration process.
        """
        self.closest_frontier_waypoint = None
        self.frontier_waypoints = np.array([])
        self._exploration_poses = []
        self._exploration_imgs = []
        self.fog_of_war_mask = np.zeros_like(self.top_down_map)
        self._agent_position = None
        self._agent_heading = None
        success = self._sample_exploration_start()
        if not success:
            return ActionIDs.STOP
        # Need to determine which action to take towards the beeline target
        action = self._decide_action(self._beeline_target)

        return action

    def _get_trajectory_mask(self, trajectory: list[np.ndarray]) -> np.ndarray:
        waypoints = [self._map_coors_to_pixel(wp) for wp in trajectory]
        # Draw lines that connect the waypoints
        path_mask = np.zeros_like(self.top_down_map)
        for i in range(len(waypoints) - 1):
            cv2.line(
                path_mask,
                tuple(waypoints[i][::-1]),
                tuple(waypoints[i + 1][::-1]),
                1,
                self._visibility_dist_in_pixels,
            )
        return path_mask

    def _save_to_dataset(self, episode: Episode) -> None:
        """
        Saves the frontier information to disk.
        """
        # 'episode_dir' path should be {self._dataset_path}/{scene_id}/{episode_id}
        scene_id = extract_scene_id(episode)
        episode_dir = osp.join(
            self._dataset_path, scene_id, f"episode_{episode.episode_id}"
        )
        if not osp.exists(episode_dir):
            os.makedirs(episode_dir, exist_ok=True)

        frontier_imgs_dir = osp.join(episode_dir, "frontier_imgs")
        self._save_frontier_images(frontier_imgs_dir)

        exploration_id = len(
            [f for f in os.listdir(episode_dir) if f.startswith("exploration_imgs_")]
        )
        exploration_imgs_dir = osp.join(
            episode_dir, f"exploration_imgs_{exploration_id}"
        )
        self._save_exploration_imgs(exploration_imgs_dir)

        episode_json = osp.join(episode_dir, f"exploration_{exploration_id}.json")
        self._save_episode_json(episode_dir, exploration_imgs_dir, episode_json)

        # Save visualization of the coverage
        h, w = self.fog_of_war_mask.shape
        coverage_img = np.zeros((h, w, 3), dtype=np.uint8)
        coverage_img[self.top_down_map == 1] = (255, 255, 255)
        coverage_img[self.fog_of_war_mask == 1] = (0, 0, 255)
        coverage_img[self._gt_fog_of_war_mask == 1] = (0, 255, 0)
        # Make the overlap of the two fogs purple
        coverage_img[
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
        for frontier_id, img in self._frontier_id_to_img.items():
            img_filename = osp.join(
                frontier_imgs_dir, f"frontier_{frontier_id:04d}.jpg"
            )
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_filename, img_bgr)

    def _save_exploration_imgs(self, exploration_imgs_dir: str) -> None:
        """
        Saves the exploration images to disk.

        Args:
            exploration_imgs_dir (str): The path to the directory to save the
                exploration images.
        """
        if not osp.exists(exploration_imgs_dir):
            os.makedirs(exploration_imgs_dir, exist_ok=True)
        for i, img in enumerate(self._exploration_imgs):
            img_filename = osp.join(exploration_imgs_dir, f"exploration_{i:04d}.jpg")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_filename, img_bgr)

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
        # Get list of jpgs from the exploration_imgs_dir
        json_data = {
            "episode_id": self._episode.episode_id,
            "scene_id": self._get_scene_id(),
            "exploration_id": int(osp.basename(exploration_imgs_dir).split("_")[-1]),
            "object_category": self._episode.object_category,
            "num_time_steps": self._step_count,
            "gt_path_poses": self._gt_path_poses,
            "exploration_coverage": float(self._exploration_coverage),
            "exploration_poses": self._exploration_poses,
            "exploration_imgs": sorted(
                [
                    osp.join(exploration_imgs_dir, f)
                    for f in glob.glob(f"{exploration_imgs_dir}/*.jpg")
                ]
            ),
            "frontiers_ids_to_imgs": {
                int(osp.basename(f).split("_")[1].split(".")[0]): f
                for f in sorted(glob.glob(f"{frontier_imgs_dir}/*.jpg"))
            },
            "timestep_to_frontiers": {
                fs.time_step: fs.to_dict() for fs in self._frontier_sets
            },
        }
        with open(episode_json, "w") as f:
            json.dump(json_data, f)

    def _get_scene_id(self) -> str:
        return os.path.basename(self._episode.scene_id).split(".")[0]

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        if self._state != State.EXPLORE:
            # These functions are only called when self._state == State.EXPLORE, but for
            # this class, we want to call it every step
            self._update_frontiers()
            self.closest_frontier_waypoint = self._get_closest_waypoint()

        if self._is_exploring:
            self._state = State.BEELINE

        action = super().get_observation(task, episode, *args, **kwargs)

        self._record_curr_pose()

        assert "observations" in kwargs, "Observations must be passed as a keyword arg"
        if not self._is_exploring:
            self._record_frontiers(kwargs["observations"]["rgb"])
        else:
            self._exploration_imgs.append(kwargs["observations"]["rgb"])

        # An episode is considered bad if the agent has timed out despite the episode
        # being feasible. However, since this sensor is always called before the map is
        # updated, we have to make sure that self._step_count is > 1
        if self._step_count == 1:
            self._bad_episode = False
        elif not self._is_exploring:  # Can stop checking once we start exploring
            feasible = self._map_measure.get_metric()["is_feasible"]
            stop_called = np.array_equal(action, ActionIDs.STOP)
            if feasible:
                self._bad_episode = not stop_called
            else:
                self._bad_episode = False
                task.is_stop_called = True
                return ActionIDs.STOP

            if feasible and stop_called:
                # Ground truth path completed; move on to exploration phase
                self._is_exploring = True
                self._gt_fog_of_war_mask = self.fog_of_war_mask.copy()

                # Reset the exploration
                action = self._reset_exploration()
                if action == ActionIDs.STOP:
                    # Could not find a valid exploration path
                    task.is_stop_called = True
                else:
                    return self.get_observation(task, episode, *args, **kwargs)
        else:
            # Exploration is active. Check if exploration has successfully completed.
            if np.array_equal(action, ActionIDs.STOP):
                # true:
                # - The length of self._exploration_poses must be within the valid range
                # - The coverage of the exploration must be within the valid range
                self._exploration_coverage = check_mask_overlap(
                    self._gt_fog_of_war_mask,
                    self.fog_of_war_mask,
                )
                self._exploration_successful = (
                    self._min_exploration_steps
                    <= len(self._exploration_poses)
                    <= self._max_exploration_steps
                    and self._min_exploration_coverage
                    <= self._exploration_coverage
                    <= self._max_exploration_coverage
                )

                if self._exploration_successful:
                    print("Exploration successful!")
                    # Save the exploration data to disk
                    self._save_to_dataset(episode)
                else:
                    print("Exploration failed!")
                    print(f"{self._exploration_coverage=}")
                    print(f"{len(self._exploration_poses)=}")
                    action = self._reset_exploration()
                    if action == ActionIDs.STOP:
                        # Could not find a valid exploration path
                        task.is_stop_called = True
                    else:
                        return self.get_observation(task, episode, *args, **kwargs)

        return action


class FrontierSet:
    def __init__(self, frontier_ids: list[int], best_id: int, time_step: int):
        self.frontier_ids = frontier_ids
        self.best_id = best_id
        self.time_step = time_step

    def to_dict(self):
        return {
            "frontier_ids": self.frontier_ids,
            "best_id": self.best_id,
        }


@dataclass
class ExplorationEpisodeGeneratorConfig(ObjNavExplorerSensorConfig):
    type: str = ExplorationEpisodeGenerator.__name__
    turn_angle: float = 30.0  # degrees
    forward_step_size: float = 0.5  # meters
    beeline_dist_thresh: float = 8  # meters
    success_distance: float = 0.1  # meters
    dataset_path: str = "data/exploration_episodes/"
    max_exploration_attempts: int = 1000
    min_exploration_steps: int = 10
    max_exploration_steps: int = 100
    min_exploration_coverage: float = 0.1
    max_exploration_coverage: float = 0.6


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.lab_sensors.exploration_episode_generator",
    group="habitat/task/lab_sensors",
    name="exploration_episode_generator",
    node=ExplorationEpisodeGeneratorConfig,
)


def extract_scene_id(episode: Episode) -> str:
    """
    Extracts the scene id from an episode.

    Args:
        episode (Episode): The episode from which to extract the scene id.

    Returns:
        str: The scene id extracted from the episode.
    """
    scene_id = os.path.basename(episode.scene_id).split(".")[0]
    return scene_id


def convert_to_frontier_filenames(
    episode_dir: str, frontier_ids: list[int]
) -> list[str]:
    """
    Converts a list of frontier ids to a list of filenames.

    Args:
        episode_dir (str): The path to the episode directory.
        frontier_ids (list[int]): A list of frontier ids.

    Returns:
        list[str]: A list of filenames corresponding to the frontier ids.
    """
    return [
        osp.join(episode_dir, f"frontier_imgs/frontier_{frontier_id:04d}.jpg")
        for frontier_id in frontier_ids
    ]


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
