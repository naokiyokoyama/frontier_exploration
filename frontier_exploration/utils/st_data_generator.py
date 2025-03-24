import glob
import hashlib
import json
import os
import os.path as osp
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import habitat_sim
import numpy as np
import quaternion as qt
import tqdm
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import calculate_meters_per_pixel

from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.general_utils import (
    calculate_angle,
    compress_int_list,
    get_polar_angle,
    habitat_to_xyz,
    images_to_video,
    sliding_window_strided,
)
from frontier_exploration.utils.sim_utils import (
    create_simulator,
    generate_path,
    load_reshaped_npy,
)
from frontier_exploration.utils.viz import (
    add_translucent_border,
    apply_color_mask,
    place_image_centered,
    rotate_image_orientation,
)


@dataclass(frozen=True)
class ImageGoal:
    position: np.ndarray
    yaw: float
    fog: np.ndarray
    rgb: np.ndarray


class STDataGenerator:
    def __init__(
        self,
        min_actions: int,
        image_goal_unobstructed_thresh: float = 0.3,
        max_attempts_image_goal: int = 100,
        meters_per_pixel: float = 1.0 / 17.0,
        min_goal_distance: float = 2.5,
        max_goal_distance: float = 15.0,
        visibility_dist: float = 5.0,
        min_tour_length: int = 200,
        max_tour_length: int = 300,
        visualize: bool = False,
    ):
        self._sim: Optional[habitat_sim.Simulator] = None
        self._fov: Optional[float] = None
        self._last_check_time: float = 0.0
        self._flag_path: str = ""
        self._min_actions: int = min_actions
        self._image_goal_unobstructed_thresh: float = image_goal_unobstructed_thresh
        self._max_attempts_image_goal: int = max_attempts_image_goal
        self._meters_per_pixel: float = meters_per_pixel
        self._visibility_dist_in_pixels: int = int(visibility_dist / meters_per_pixel)
        self._min_goal_distance: float = min_goal_distance
        self._max_goal_distance: float = max_goal_distance
        self._min_tour_length: int = min_tour_length
        self._max_tour_length: int = max_tour_length
        self._should_visualize: bool = visualize

    def load_scene(self, scene_path: str) -> None:
        if self._sim is not None:
            self._sim.close()
        self._sim = create_simulator(scene_path, agent_radius=0.09)
        self._fov = float(self._sim._sensors["rgb_sensor"]._spec.hfov)  # noqa

    def should_stop(self, output_dir: str, num_episodes: int) -> bool:
        """
        True if any of the following conditions are met:
            1. The requested amount of episodes has already been generated
            2. The exploration has already been completed or found to have been
                impossible to generate episodes for.
        """
        if time.time() - self._last_check_time < 0.5:
            return False
        self._last_check_time = time.time()

        # Check if the exploration has already been completed, or if this exploration
        # has been marked as impossible to generate episodes for
        mp4_count = self._get_mp4_count(output_dir)
        if mp4_count >= num_episodes:
            print(f"{mp4_count} of {num_episodes} episodes have been generated.")
            return True
        if osp.exists(osp.join(output_dir, ".impossible.flag")):
            print("This exploration has been marked as impossible.")
            return True

        return False

    @staticmethod
    def _get_mp4_count(output_dir: str) -> int:
        st_mp4s = glob.glob(output_dir + "/st_*.mp4")
        return len(st_mp4s)

    def generate_episodes(self, exploration_file_path: str, num_episodes: int) -> None:
        # Find the start pose of the exploration by locating the JSON file
        output_dir = osp.dirname(exploration_file_path)
        exploration_start = self._get_exploration_start(output_dir)

        # Get the top-down map from the simulator
        top_down_map = np.ascontiguousarray(
            self._sim.pathfinder.get_topdown_view(
                meters_per_pixel=self._meters_per_pixel, height=exploration_start[1]
            ).astype(np.uint8)
        )
        exploration_masks = load_reshaped_npy(exploration_file_path)
        for _ in range(num_episodes):
            generated = False
            for _ in range(10000):
                if self.should_stop(output_dir, num_episodes):
                    return
                generated = self._generate_episode(
                    exploration_masks,
                    exploration_start,
                    top_down_map,
                    output_dir,
                    num_episodes,
                )
                if generated:
                    break

            if not generated:
                print("Could not find a valid episode.")
                with open(osp.join(output_dir, ".impossible.flag"), "w"):
                    pass

    def _generate_episode(
        self,
        exploration_masks: np.ndarray,
        exploration_start: np.ndarray,
        top_down_map: np.ndarray,
        output_dir: str,
        num_episodes: int,
    ) -> bool:
        def sample_image_goals(
            destination: np.ndarray,
        ) -> Optional[Tuple[np.ndarray, List[ImageGoal]]]:
            def check_goal_fogs(p: List[np.ndarray]) -> Optional[List[ImageGoal]]:
                im_goals = []
                for i in p:
                    img_goal = self._sample_image_goal(i, top_down_map)
                    if img_goal is None:
                        return None
                    im_goals.append(img_goal)
                return im_goals

            def get_goal_pos(
                dst: np.ndarray, curr_goals: List[np.ndarray]
            ) -> Optional[np.ndarray]:
                xy_list = [habitat_to_xyz(i)[:2] for i in curr_goals]
                dst_xy = habitat_to_xyz(dst)[:2]
                for _ in range(10):
                    goal_pt = self._sample_point(exploration_start)
                    if goal_pt is None:
                        continue
                    if not (
                        self._min_goal_distance
                        <= np.linalg.norm(dst - goal_pt)
                        <= self._max_goal_distance
                    ):
                        continue

                    success = True
                    for xy in xy_list:
                        goal_xy = habitat_to_xyz(goal_pt)[:2]
                        if calculate_angle(xy, dst_xy, goal_xy) <= np.radians(20):
                            success = False
                            break
                        if np.linalg.norm(xy - goal_xy) < self._min_goal_distance:
                            success = False
                            break

                    if success:
                        return goal_pt

                return None

            def get_goal_set(dst: np.ndarray) -> Optional[List[np.ndarray]]:
                valid_goals = []
                for _ in range(4):
                    goal = get_goal_pos(dst, valid_goals)
                    if goal is None:
                        return None
                    valid_goals.append(goal)
                return valid_goals

            goal_set = get_goal_set(destination)
            if goal_set is not None:
                img_goals = check_goal_fogs(goal_set)
                if img_goals is not None:
                    return destination, img_goals

        # Sample four image goals in the environment until valid ones are found
        print("Finding image goals...")
        destination = self._sample_point(exploration_start)
        if destination is None:
            return False
        for _ in tqdm.trange(5):
            if self.should_stop(output_dir, num_episodes):
                return False
            dst_pos_and_image_goals = sample_image_goals(destination)
            if dst_pos_and_image_goals is None:
                continue
            dst_pos, image_goals = dst_pos_and_image_goals
            poses = self._sample_path(exploration_start, dst_pos)
            if not poses:
                print("Could not find a valid path.")
                continue
            tour_dict = self._check_image_goals(
                image_goals, poses, exploration_masks, top_down_map
            )
            if not tour_dict:
                print("Image goals are not valid.")
                continue
            # Generate an MP4 of the agent moving from start to end
            frames = self._render_trajectory(poses)
            if self.should_stop(output_dir, num_episodes):
                return False
            self._save_info(image_goals, poses, frames, tour_dict, output_dir)
            return True

        return False

    def _get_exploration_start(self, exploration_dir: str) -> np.ndarray:
        return self._get_exploration(exploration_dir)[0][:3]

    @staticmethod
    def _get_exploration(exploration_dir: str) -> np.ndarray:
        exploration_file = osp.join(osp.dirname(exploration_dir), "exploration_0.json")
        if not osp.exists(exploration_file):
            raise FileNotFoundError(f"Exploration file '{exploration_file}' not found.")
        with open(exploration_file, "r") as f:
            data = json.load(f)

        return np.array(data["exploration_poses"])

    def _sample_point(self, exploration_start: np.ndarray) -> Optional[np.ndarray]:
        for _ in range(1000):
            pt = self._sim.pathfinder.get_random_navigable_point()
            if abs(pt[1] - exploration_start[1]) > 0.7:
                continue
            path = habitat_sim.nav.ShortestPath()
            path.requested_start = exploration_start
            path.requested_end = pt
            if self._sim.pathfinder.find_path(path):
                return pt

    def _sample_path(
        self, exploration_start: np.ndarray, destination: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        print("Sampling path...")
        for _ in tqdm.trange(1000):
            start = self._sample_point(exploration_start)
            if start is None:
                continue
            end = destination
            path = habitat_sim.nav.ShortestPath()
            path.requested_start = start
            path.requested_end = end
            assert self._sim.pathfinder.find_path(path)
            poses = generate_path(np.array(path.points))
            if len(poses) >= self._min_actions:
                return poses[-self._min_actions :]

        return []

    def _sample_image_goal(
        self, position: np.ndarray, top_down_map: np.ndarray
    ) -> Optional[ImageGoal]:
        """
        Sample an image goal that is reachable from the exploration start position.

        Args:
            position (np.ndarray): The position of the image goal
            top_down_map (np.ndarray): The map of the scene at the correct elevation.

        Returns:
            Optional[ImageGoal]: An ImageGoal dataclass containing position, fog mask, and RGB
                image. None if a valid image goal could not be found.
        """
        # Calculate total FOV pixels once for comparison
        total_fov_pixels = np.count_nonzero(
            reveal_fog_of_war(
                top_down_map=np.ones_like(top_down_map),  # fully navigable
                current_fog_of_war_mask=np.zeros_like(top_down_map),  # no exploration
                current_point=np.array(top_down_map.shape) // 2,
                current_angle=0.0,
                fov=self._fov,
                max_line_len=self._visibility_dist_in_pixels,
            )
        )
        # Convert goal position to pixel coordinates for FOV calculation
        sampled_position_px = self._3d_to_map_px(position, top_down_map)
        for yaw_attempt in range(50):
            # Generate a random angle
            rand_angle = np.random.uniform(0, 2 * np.pi)
            goal_rotation = qt.quaternion(
                np.cos(rand_angle / 2), 0, np.sin(rand_angle / 2), 0
            )
            # Calculate FOV visibility for this angle
            yaw = get_polar_angle(goal_rotation)  # noqa
            curr_fov = reveal_fog_of_war(
                top_down_map=top_down_map,
                current_fog_of_war_mask=np.zeros_like(top_down_map),
                current_point=sampled_position_px,
                current_angle=yaw,
                fov=self._fov,
                max_line_len=self._visibility_dist_in_pixels,
            )
            if (
                np.count_nonzero(curr_fov) / total_fov_pixels
                > self._image_goal_unobstructed_thresh
            ):
                # Set the agent state to the sampled_position and goal_rotation
                self._sim.get_agent(0).set_state(
                    habitat_sim.AgentState(position=position, rotation=goal_rotation)
                )
                return ImageGoal(
                    position=position,
                    yaw=yaw,
                    fog=curr_fov,
                    rgb=self._sim.get_sensor_observations()["rgb_sensor"][..., :3],
                )

    def _3d_to_map_px(
        self, coord_3d: np.ndarray, top_down_map: np.ndarray
    ) -> np.ndarray:
        return np.array(
            maps.to_grid(
                float(coord_3d[2]),
                float(coord_3d[0]),
                (top_down_map.shape[0], top_down_map.shape[1]),
                sim=self._sim,
            )
        )

    def _check_image_goals(
        self,
        image_goals: List[ImageGoal],
        poses: List[Tuple[np.ndarray, float]],
        exploration_masks: np.ndarray,
        top_down_map: np.ndarray,
    ) -> Dict[int, Tuple[Union[Tuple[int, int], int], ...]]:
        """
        The ImageGoals are valid if:
        1. None of them are too close to each other
        2. None of them are visible to the agent's final pose
        3. All or all but one of them intersect sufficiently with the exploration mask
        """
        # Check if the image goals are visible from the final pose
        yaw = poses[-1][1]
        agent_heading = get_polar_angle(
            qt.quaternion(0, np.sin(yaw / 2), 0, np.cos(yaw / 2))  # noqa
        )
        final_fog = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=np.zeros_like(top_down_map),  # no exploration
            current_point=self._3d_to_map_px(poses[-1][0], top_down_map),
            current_angle=agent_heading,
            fov=self._fov,
            max_line_len=self._visibility_dist_in_pixels * 2,
        )
        for goal in image_goals:
            pos_px = self._3d_to_map_px(goal.position, top_down_map)
            if final_fog[pos_px[0], pos_px[1]]:
                return {}

        # Check if the image goals intersect with exploration mask within a valid
        # window size
        goal_fogs = np.stack([goal.fog for goal in image_goals], axis=0)
        overlaps = compute_mask_overlap(goal_fogs, exploration_masks, threshold=0.2)
        min_tour = min(self._min_tour_length, exploration_masks.shape[0])
        max_tour = min(self._max_tour_length, exploration_masks.shape[0])
        result = {}
        for window_size in range(min_tour, max_tour + 1):
            windows = sliding_window_strided(overlaps, window_size)
            valid_windows = windows.any(axis=1).all(axis=1)
            if not valid_windows.any():
                continue
            valid_window_indices = np.where(valid_windows)[0]
            result[window_size] = compress_int_list(valid_window_indices.tolist())

        return result

    def _render_trajectory(
        self, path: List[Tuple[np.ndarray, float]]
    ) -> List[np.ndarray]:
        agent = self._sim.get_agent(0)
        frames = []
        for pos, rot in path:
            # Convert yaw to quaternion
            rot = np.array([0, np.sin(rot / 2), 0, np.cos(rot / 2)])
            agent.set_state(habitat_sim.AgentState(position=pos, rotation=rot))
            observations = self._sim.get_sensor_observations()
            frames.append(observations["rgb_sensor"][..., :3])

        return frames

    def _save_info(
        self,
        image_goals: List[ImageGoal],
        poses: List[Tuple[np.ndarray, float]],
        frames: List[np.ndarray],
        tour_dict: Dict[int, Tuple[Union[Tuple[int, int], int], ...]],
        output_dir: str,
    ) -> None:
        # Generate json data representing relative coordinate info
        coords = self._compute_relative_coords(image_goals, poses)
        coords_str = self._hash_coords(coords)

        # Save video
        mp4_path = osp.join(output_dir, f"st_{coords_str}.mp4")
        images_to_video(
            [
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for frame in [i.rgb for i in image_goals] + frames
            ],
            mp4_path,
        )

        # Save tour and coordinates to a pickle file
        with open(osp.join(output_dir, f"st_{coords_str}.pkl"), "wb") as f:
            pickle.dump((tour_dict, coords), f)  # noqa

        if self._should_visualize:
            self._visualize(image_goals, poses, frames, tour_dict, output_dir)

    @staticmethod
    def _compute_relative_coords(
        image_goals: List[ImageGoal], poses: List[Tuple[np.ndarray, float]]
    ) -> List[Tuple[float, float]]:
        result = []
        source, yaw = poses[-1]
        yaw = convert_heading(yaw)
        source = habitat_to_xyz(source)[:2]
        for goal in image_goals:
            target = habitat_to_xyz(goal.position)[:2]
            coord = get_relative_location(source, target, yaw)
            result.append((-float(coord[0]), float(coord[1])))
        return result

    @staticmethod
    def _hash_coords(coords: List[Tuple[float, float]]) -> str:
        # Convert the tuples to a string representation with high precision
        tuple_str = ""
        for tup in coords:
            tuple_str += f"{tup[0]:.16f},{tup[1]:.16f};"

        # Create hash from the string
        hash_obj = hashlib.sha256(tuple_str.encode())

        # Return the hexadecimal representation of the hash
        return hash_obj.hexdigest()[:6]

    def _visualize(
        self,
        image_goals: List[ImageGoal],
        poses: List[Tuple[np.ndarray, float]],
        frames: List[np.ndarray],
        tour_dict: Dict[int, Tuple[Union[Tuple[int, int], int], ...]],
        output_dir: str,
    ) -> None:
        """
        Create a visualization of the top-down map with image goals and agent poses.

        Args:
            image_goals: List of ImageGoal objects to visualize
            poses: List of agent (position, rotation) tuples
            frames: List of frames rendered from the poses
            output_dir: Output directory
        """
        coords = self._compute_relative_coords(image_goals, poses)

        # Create an RGB top down map for visualization
        meters_per_pixel = calculate_meters_per_pixel(map_resolution=640, sim=self._sim)
        top_down_map = np.ascontiguousarray(
            self._sim.pathfinder.get_topdown_view(
                meters_per_pixel=meters_per_pixel, height=poses[0][0][1]
            ).astype(np.uint8)
        )
        viz_img = np.stack([top_down_map] * 3, axis=-1, dtype=np.uint8)
        viz_img[top_down_map > 0] = (255, 255, 255)
        x, y, w, h = cv2.boundingRect((top_down_map == 0).astype(np.uint8))
        y_min, y_max = max(0, y - 15), min(top_down_map.shape[0], y + h + 15)
        x_min, x_max = max(0, x - 15), min(top_down_map.shape[1], x + w + 15)

        # Draw the exploration fogs
        current_fog = np.zeros_like(top_down_map)
        visibility_dist_in_pixels = int(4.5 / meters_per_pixel)
        exploration_poses = self._get_exploration(output_dir)
        random_window = random.choice(list(tour_dict.keys()))
        start = random.choice(tour_dict[random_window])
        end = start + random_window
        exploration_poses = exploration_poses[start:end]
        for exp_pose in exploration_poses:
            position = exp_pose[:3]
            angle = exp_pose[3]
            current_fog = reveal_fog_of_war(
                top_down_map=top_down_map,
                current_fog_of_war_mask=current_fog,
                current_point=self._3d_to_map_px(position, top_down_map),
                current_angle=angle,
                fov=self._fov,
                max_line_len=visibility_dist_in_pixels,
            )
        viz_img[current_fog > 0] = (180, 180, 180)

        RED = (0, 0, 255)  # BGR format: B=0, G=0, R=255
        ORANGE = (0, 165, 255)  # BGR format: B=0, G=165, R=255
        YELLOW = (0, 255, 255)  # BGR format: B=0, G=255, R=255
        GREEN = (0, 255, 0)  # BGR format: B=0, G=255, R=0
        BLUE = (255, 0, 0)  # BGR format: B=255, G=0, R=0
        PURPLE = (128, 0, 128)  # BGR format: B=128, G=0, R=128
        PINK = (255, 0, 255)  # BGR format: B=255, G=0, R=255
        BLACK = (0, 0, 0)  # BGR format: B=0, G=0, R=0

        # Draw each of the image goals on the visualization
        image_goal_colors = [RED, ORANGE, YELLOW, GREEN]
        for i in range(4):
            position, yaw = poses[-1]
            position_px = self._3d_to_map_px(position, top_down_map)
            angle = convert_heading(yaw)
            # Create a rotation matrix
            rotation_matrix = np.array(
                [
                    [np.sin(angle), np.sin(angle - np.pi / 2)],
                    [np.cos(angle), np.cos(angle - np.pi / 2)],
                ]
            )

            # Scale the coordinates by meters_per_pixel
            scaled_coords = np.array([coords[i][0], coords[i][1]]) / meters_per_pixel

            # Apply rotation and translation in one step
            goal_position_px = (rotation_matrix @ scaled_coords).astype(
                np.int32
            ) + position_px[::-1]

            cv2.circle(viz_img, goal_position_px, 10, PINK, -1)
            cv2.circle(viz_img, goal_position_px, 10, BLACK, 2)

            # Draw the image goal's fog on the visualization
            fog = reveal_fog_of_war(
                top_down_map=top_down_map,
                current_fog_of_war_mask=np.zeros_like(top_down_map),
                current_point=self._3d_to_map_px(image_goals[i].position, top_down_map),
                current_angle=image_goals[i].yaw,
                fov=self._fov,
                max_line_len=self._visibility_dist_in_pixels,
            )
            viz_img = apply_color_mask(viz_img, fog, image_goal_colors[i], opacity=0.75)

        # Draw the fog of the last pose on the visualization in PURPLE
        position, yaw = poses[-1]
        angle = convert_heading(yaw)
        last_fog = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=np.zeros_like(top_down_map),
            current_point=self._3d_to_map_px(position, top_down_map),
            current_angle=angle,
            fov=self._fov,
            max_line_len=self._visibility_dist_in_pixels,
        )
        viz_img = apply_color_mask(viz_img, last_fog, PURPLE, opacity=0.5)

        all_maps: List[np.ndarray] = []
        current_fog = np.zeros_like(top_down_map)
        for idx in range(len(poses)):
            if idx != 0:
                position, yaw = poses[idx - 1]
                # Generate a fog for the current pose
                angle = convert_heading(yaw)
                current_point = self._3d_to_map_px(position, top_down_map)
                current_fog = reveal_fog_of_war(
                    top_down_map=top_down_map,
                    current_fog_of_war_mask=current_fog,
                    current_point=current_point,
                    current_angle=angle,
                    fov=self._fov,
                    max_line_len=self._visibility_dist_in_pixels,
                )

                # Draw the current fog on the visualization in BLUE
                img = apply_color_mask(viz_img.copy(), current_fog, BLUE, opacity=0.5)

                # Draw the agent's current position and heading
                agent_size = 10
                cv2.circle(img, current_point[::-1], agent_size, (255, 192, 15), -1)

                heading_end_pt = (
                    agent_size * 1.4 * np.array([np.sin(angle), np.cos(angle)])
                ) + current_point[::-1]

                # Draw a line from the current position showing the current_angle
                cv2.line(
                    img,
                    current_point[::-1],
                    (int(heading_end_pt[0]), int(heading_end_pt[1])),
                    (0, 0, 0),
                    max(1, agent_size // 4),
                )
            else:
                img = viz_img

            # Crop and resize the visualization
            img = img[y_min:y_max, x_min:x_max]
            img = rotate_image_orientation(img, portrait_mode=False)
            img = place_image_centered(img, 480, 640, bg_color=(0, 0, 0))

            all_maps.append(img)

        template = np.zeros((480 * 2, 640 * 3, 3), dtype=np.uint8)
        goal_rgbs = [
            add_translucent_border(
                cv2.cvtColor(i.rgb, cv2.COLOR_BGR2RGB), thickness=40, color=color
            )
            for i, color in zip(image_goals, image_goal_colors)
        ]
        for coord, rgb in zip(coords, goal_rgbs):
            # Add the coordinate as text
            cv2.putText(
                rgb,
                f"({coord[0]:.1f}, {coord[1]:.1f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb,
                f"({coord[0]:.1f}, {coord[1]:.1f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        template[:, 640:, :] = np.vstack(
            [np.hstack(goal_rgbs[:2]), np.hstack(goal_rgbs[2:])]
        )
        all_frames = []
        for frame, map_subframe in zip(frames, all_maps):
            full_frame = template.copy()
            full_frame[:480, :640] = map_subframe
            full_frame[480:, :640, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(full_frame)

        scene_id = output_dir.split("/episode_")[0].split("/")[-1]
        ep_id = output_dir.split("/episode_")[-1].split("/")[0]
        coords_str = self._hash_coords(coords)
        output_path = f"st_viz_videos/st_{scene_id}_{ep_id}_{coords_str}_viz.mp4"
        if not osp.isdir("st_viz_videos"):
            os.mkdir("st_viz_videos")
        images_to_video(all_frames, output_path)


def convert_heading(heading: float) -> float:
    return get_polar_angle(
        qt.quaternion(np.cos(heading / 2), 0, np.sin(heading / 2), 0)
    )


def get_relative_location(source: np.ndarray, target: np.ndarray, yaw: float):
    """
    Calculate the relative location of the target from the source position with given
    heading.

    Args:
        source (np.ndarray): Source position as (x, y) coordinates, shape (2,)
        target (np.ndarray): Target position as (x, y) coordinates, shape (2,)
        yaw (float): Heading angle in radians

    Returns:
        np.ndarray: Relative location of target in the source's coordinate frame, shape
                   (2,)
    """
    # Calculate the vector from source to target in global coordinates
    vector = target - source

    # Create rotation matrix for the yaw angle
    rotation_matrix = np.array(
        [[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]]
    )

    # Apply rotation to transform the vector to local coordinates
    relative_location = np.dot(rotation_matrix, vector)

    return relative_location


def compute_mask_overlap(
    masks1: np.ndarray, masks2: np.ndarray, threshold: float = 0.2
) -> np.ndarray:
    """
    Compute overlap between two sets of binary masks.

    Args:
        masks1 (np.ndarray): Binary array of shape (N, H, W)
        masks2 (np.ndarray): Binary array of shape (M, H, W)
        threshold (float, optional): Minimum percentage of overlap required, default 0.2
                                    (20%)

    Returns:
        np.ndarray: Binary array of shape (M, N) where element (m, n) is True if:
                    1. The n-th mask in masks1 has at least one set pixel
                    2. The percentage of set pixels in n-th mask of masks1 that overlap
                       with set pixels in m-th mask of masks2 is above threshold
    """
    # Ensure inputs are binary
    masks1 = masks1.astype(bool)
    masks2 = masks2.astype(bool)

    # Get the dimensions
    N, H, W = masks1.shape
    M = masks2.shape[0]

    # Count set pixels in each mask from masks1
    mask1_areas = masks1.reshape(N, -1).sum(axis=1)  # Shape: (N,)

    # Create a matrix to store results
    overlap_matrix = np.zeros((M, N), dtype=bool)

    # Only proceed with masks that have non-zero areas
    valid_mask_indices = np.where(mask1_areas > 0)[0]

    if len(valid_mask_indices) > 0:
        # Reshape masks1 for vectorized computation
        masks1_valid = masks1[valid_mask_indices].reshape(
            len(valid_mask_indices), 1, H, W
        )

        # Compute intersection in a vectorized way
        # Broadcast masks1_valid to (valid_N, M, H, W) and masks2 to (1, M, H, W)
        intersection = masks1_valid & masks2.reshape(1, M, H, W)

        # Sum over the spatial dimensions to get intersection area
        intersection_areas = intersection.reshape(len(valid_mask_indices), M, -1).sum(
            axis=2
        )  # Shape: (valid_N, M)

        # Calculate percentage of overlap
        # Reshape mask1_areas for broadcasting
        mask1_areas_valid = mask1_areas[valid_mask_indices].reshape(
            -1, 1
        )  # Shape: (valid_N, 1)

        # Compute percentage of masks1 pixels that overlap with masks2
        overlap_percentage = (
            intersection_areas / mask1_areas_valid
        )  # Shape: (valid_N, M)

        # Apply threshold
        valid_overlaps = overlap_percentage > threshold  # Shape: (valid_N, M)

        # Store results in the appropriate indices
        # Need to transpose since we need output of shape (M, N)
        overlap_matrix[:, valid_mask_indices] = valid_overlaps.T

    return overlap_matrix


def max_consecutive_true(bool_array: np.ndarray) -> int:
    """
    Find the maximum number of consecutive True values in a boolean array.

    Args:
        bool_array (np.ndarray): Boolean array of shape (N,)

    Returns:
        int: The length of the longest consecutive sequence of True values
    """
    # Handle empty array case
    if len(bool_array) == 0:
        return 0

    # If there are no True values at all, return 0
    if not np.any(bool_array):
        return 0

    # Find positions where values change
    transitions = np.where(np.diff(np.concatenate(([False], bool_array, [False]))))[0]

    # Check if transitions has an odd length (which would cause the error)
    if len(transitions) % 2 != 0:
        # This indicates the array starts or ends with True values
        # Add proper handling for this case
        transitions = np.concatenate((transitions, [len(bool_array)]))

    # Calculate lengths of all True sequences
    lengths = transitions[1::2] - transitions[::2]

    # Return the maximum length
    return np.max(lengths) if len(lengths) > 0 else 0


def main(
    exploration_episode_dir: str, num_episodes: int, random_seed: int, visualize: bool
):
    exploration_episode_dir = osp.abspath(exploration_episode_dir)
    scene_name = exploration_episode_dir.split("/episode_")[0].split("/")[-1]
    scene_glb = next(
        glob.iglob(
            f"data/scene_datasets/hm3d/train/*{scene_name}/{scene_name}*.basis.glb",
            recursive=True,
        )
    )
    try:
        exploration_npy = next(
            glob.iglob(exploration_episode_dir + "/**/exploration*.npy", recursive=True)
        )
    except StopIteration:
        return
    generator = STDataGenerator(min_actions=20, visualize=visualize)

    output_dir = osp.dirname(exploration_npy)
    if generator.should_stop(output_dir, num_episodes):
        return

    generator.load_scene(scene_glb)

    if random_seed != -1:
        np.random.seed(random_seed)
        generator._sim.seed(random_seed)
        random.seed(random_seed)
    generator.generate_episodes(exploration_npy, num_episodes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exploration-episode",
        type=str,
        help="Path to the exploration episode directory",
    )
    parser.add_argument(
        "--exploration-dir",
        type=str,
        help="Path to the exploration directory containing all scenes",
    )
    parser.add_argument(
        "--num-episodes", type=int, required=True, help="Number of episodes to generate"
    )
    parser.add_argument("-r", "--random-seed", type=int, default=-1, help="Random seed")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize data")
    args = parser.parse_args()

    assert (
        args.exploration_episode or args.exploration_dir
    ), "Either --exploration-episode or --exploration-dir must be provided"
    assert not (
        args.exploration_episode and args.exploration_dir
    ), "Only one of --exploration-episode or --exploration-dir can be provided"

    if args.exploration_episode:
        main(
            args.exploration_episode,
            args.num_episodes,
            args.random_seed,
            args.visualize,
        )
    else:
        scenes = glob.glob(args.exploration_dir + "/*/")
        random.shuffle(scenes)
        for scene in scenes:
            episodes = glob.glob(scene + "/episode_*/")
            random.shuffle(episodes)
            for episode in episodes:
                main(
                    episode,
                    args.num_episodes,
                    args.random_seed,
                    args.visualize,
                )
