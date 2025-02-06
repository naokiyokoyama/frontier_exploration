import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
from numba import njit

from frontier_exploration.utils.frontier_utils import closest_line_segment
from frontier_exploration.utils.general_utils import wrap_heading


class FrontierInfo:
    def __init__(
        self,
        agent_pose: list[float],  # in Habitat coordinates
        camera_position_px: tuple[int, int],  # in pixel coordinates
        frontier_position: np.ndarray,  # in Habitat coordinates
        frontier_position_px: tuple[int, int],  # in pixel coordinates
        single_fog_of_war: np.ndarray,
        rgb_img: np.ndarray,
    ) -> None:
        self.agent_pose = agent_pose
        self.camera_position_px = np.array(camera_position_px)
        self.camera_yaw = agent_pose[3]
        self.frontier_position = frontier_position
        self.frontier_position_px = np.array(frontier_position_px)
        self.single_fog_of_war = single_fog_of_war
        self.rgb_img = rgb_img

    @property
    def position_tuple(self):
        return tuple(self.frontier_position_px)

    @property
    def agent_pose_tuple(self):
        return tuple(self.agent_pose)

    def to_dict(self, frontier_id: int, frontier_imgs_dir: str):
        return {
            frontier_id: {
                "agent_pose": self.agent_pose,
                "frontier_position_px": self.frontier_position_px.tolist(),
                "frontier_position": self.frontier_position.tolist(),
                "frontier_img_path": osp.join(
                    frontier_imgs_dir, f"frontier_{frontier_id:04d}.jpg"
                ),
            }
        }

    def is_overlapping(self, other: "FrontierInfo") -> bool:
        """
        Two frontiers overlap if the difference between their two camera yaws is less
        than 45 degrees and the overlap between their single fog of wars is greater
        than 50%.
        """
        assert self.single_fog_of_war.shape == other.single_fog_of_war.shape

        yaw_diff = abs(wrap_heading(self.camera_yaw - other.camera_yaw))
        if yaw_diff > np.deg2rad(45):
            return False

        nonzero1 = np.count_nonzero(self.single_fog_of_war)
        nonzero2 = np.count_nonzero(other.single_fog_of_war)

        # Count overlapping non-zero pixels
        overlap = np.count_nonzero(
            np.logical_and(self.single_fog_of_war != 0, other.single_fog_of_war != 0)
        )

        # Calculate percentage overlap
        if nonzero1 == 0 or nonzero2 == 0:
            percentage_overlap = 0
        else:
            percentage_overlap = max(overlap / nonzero1, overlap / nonzero2)
        return percentage_overlap > 0.5


def filter_frontiers(
    frontier_datas: List[FrontierInfo], boundary_contour: np.ndarray, gt_idx: int = -1
) -> Tuple[List[int], Dict[int, int]]:
    filtered_finfos: List[Tuple[int, FrontierInfo]] = []
    bad_idx_to_good_idx: Dict[int, int] = {}

    for idx, curr_finfo in enumerate(frontier_datas):
        # For each frontier, check if any frontier so far in the filtered list overlaps
        # with it. If not, add it to the filtered list. If it does, for all overlapping
        # frontiers, keep the one with the best boundary angle, and, importantly, remove
        # the other overlapping frontiers from the filtered list and indices to keep.
        is_overlapping = [curr_finfo.is_overlapping(f) for _, f in filtered_finfos]

        if not any(is_overlapping):
            # Current frontier passes; add it to the filtered list
            filtered_finfos.append((idx, curr_finfo))
            continue

        # Current frontier overlaps with at least one frontier in the filtered list;
        # remove these overlapping frontiers from the filtered list
        overlapping_finfos = [
            (i, f) for j, (i, f) in enumerate(filtered_finfos) if is_overlapping[j]
        ]
        filtered_finfos = [
            (i, f) for j, (i, f) in enumerate(filtered_finfos) if not is_overlapping[j]
        ]

        # Add current frontier to the overlapping ones
        overlapping_finfos.append((idx, curr_finfo))

        # Keep the best frontier among the overlapping ones
        overlapping_inds = [i for i, _ in overlapping_finfos]
        if gt_idx in overlapping_inds:
            # The ground truth is one of the overlapping ones; keep it
            best_finfo = (gt_idx, frontier_datas[gt_idx])
        else:
            best_finfo = overlapping_finfos[
                identify_best_frontier(
                    [i for _, i in overlapping_finfos], boundary_contour
                )
            ]

        # Add the best frontier to the filtered list
        filtered_finfos.append(best_finfo)

        # Map the bad indices to the good ones
        for i, _ in overlapping_finfos:
            if i != best_finfo[0]:
                bad_idx_to_good_idx[i] = best_finfo[0]

        # If any of the bad indices were previously good indices, update them
        # final_bad_inds = [i for i, _ in overlapping_finfos if i != best_finfo[0]]
        # new_dict = {}
        # for k, v in bad_idx_to_good_idx.items():
        #     new_dict[k] = best_finfo[0] if v in final_bad_inds else v
        # bad_idx_to_good_idx = new_dict

    inds_to_keep = [i for i, _ in filtered_finfos]

    return sorted(inds_to_keep), bad_idx_to_good_idx


def identify_best_frontier(
    frontier_datas: List[FrontierInfo], boundary_contour: np.ndarray
) -> int:
    """
    Given a list of FrontierInfo, return the whose 'boundary angle' is closest to 90.
    The boundary angle is the angle between the line connecting the camera position and
    the frontier position, and the line segment within the boundary contour that is
    closest to the frontier position.
    """
    assert len(frontier_datas) > 0
    if np.count_nonzero(boundary_contour) <= 2:
        return 0

    best_frontier_idx = None
    best_angle = np.inf

    for idx, frontier_data in enumerate(frontier_datas):
        angle = get_boundary_angle(frontier_data, boundary_contour)
        angle = abs(angle - np.pi / 2)
        if angle < best_angle:
            best_frontier_idx = idx
            best_angle = angle

    return best_frontier_idx


def get_boundary_angle(
    frontier_data: FrontierInfo, boundary_contour: np.ndarray
) -> float:
    points = boundary_contour.reshape(-1, 2)
    line_segments = np.concatenate(
        [points[:-1, np.newaxis], points[1:, np.newaxis]], axis=1
    )
    try:
        closest_segment = closest_line_segment(
            frontier_data.camera_position_px, line_segments
        )[0]
    except BaseException as e:
        print(f"{boundary_contour = }")
        raise e
    cam_to_frontier_segment = np.array(
        [frontier_data.camera_position_px, frontier_data.frontier_position_px]
    )
    angle = min_angle_between_lines(cam_to_frontier_segment, closest_segment)

    return angle


@njit
def min_angle_between_lines(line1, line2):
    # Ensure inputs are correct shape
    assert len(line1) == len(line2) == 2
    assert len(line1[0]) == len(line1[1]) == len(line2[0]) == len(line2[1]) == 2

    # Calculate vectors for each line
    vec1 = line1[1] - line1[0]
    vec2 = line2[1] - line2[0]

    # Calculate dot product
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]

    # Calculate magnitudes
    mag1 = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    mag2 = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

    # Check for zero-length vectors
    if mag1 == 0 or mag2 == 0:
        return np.pi / 2

    # Calculate cosine of the angle
    cos_angle = dot_product / (mag1 * mag2)

    # Clamp cosine to [-1, 1] to avoid errors caused by floating point precision
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # Calculate the angle in radians
    angle = np.arccos(cos_angle)

    # Return the minimum angle (acute angle)
    return min(angle, np.pi - angle)
