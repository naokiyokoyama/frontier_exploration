from typing import List, Optional

import cv2
import numpy as np

from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.general_utils import (
    calculate_perpendicularity,
    interpolate_line,
)


def get_action(
    frontier_segments: List[np.ndarray],
    obstacle_map: np.ndarray,
    camera_pos: np.ndarray,
    camera_yaw: np.ndarray,
    fov: float,
    max_line_len: int,
    turn_angle: float,
) -> Optional[bool]:
    assert (np.pi * 2 / turn_angle) % 2 == 0, "Turn angle must divide 360 evenly"
    max_line_len = int(max_line_len * 0.75)
    # 1. Interpolate to make dense versions of the frontier segments
    dense_frontier_segments = [  # List of shape (X, 2,) arrays
        interpolate_line(f_seg) for f_seg in frontier_segments
    ]

    # 2. Compute the distance between the camera and each point in the dense frontier
    # segments
    all_seg_pts = np.vstack(dense_frontier_segments)
    dists = np.linalg.norm(all_seg_pts - camera_pos, axis=1)
    within_dist = dists < max_line_len * 0.98
    # Return None if no frontier segments are within the max line length
    if not within_dist.any():
        return None

    # 3. Compute the dot product between the camera and the ends of each frontier
    seg_ends = np.array([f_seg[[0, -1]] for f_seg in dense_frontier_segments])
    dot_prod = calculate_perpendicularity(camera_pos, seg_ends)  # shape: (S,)
    is_visible = dot_prod < np.cos(np.radians(8))  # 8 degrees
    # Return None if no frontier segments are visible enough
    if not is_visible.any():
        return None

    # 4. Compute the left and right fogs of war
    num_turns = int(np.pi * 2 / turn_angle)
    cropped_obstacle_map = obstacle_map[
        camera_pos[0] - max_line_len : camera_pos[0] + max_line_len,
        camera_pos[1] - max_line_len : camera_pos[1] + max_line_len,
    ]
    if cropped_obstacle_map.size == 0:
        return None

    all_fogs = [
        cv2.erode(
            reveal_fog_of_war(
                top_down_map=cropped_obstacle_map,
                current_fog_of_war_mask=np.zeros_like(cropped_obstacle_map),
                current_point=np.array([max_line_len, max_line_len]),
                current_angle=camera_yaw - i * turn_angle,
                fov=fov,
                max_line_len=max_line_len,
            ),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        for i in range(1, num_turns)
    ]

    H, W = cropped_obstacle_map.shape
    N = num_turns // 2
    left_right_fogs = np.empty((2, N, H, W), dtype=np.int32)
    for idx in range(N):
        left_right_fogs[0, idx] = all_fogs[idx]
        left_right_fogs[1, idx] = all_fogs[num_turns - 2 - idx]

    # 5. Create binary mask for each frontier segment and stack them
    S = len(frontier_segments)
    f_seg_masks = np.zeros((S, H, W), dtype=np.int32)
    start_idx = 0
    total_counts = []
    offset = np.array([camera_pos[0] - max_line_len, camera_pos[1] - max_line_len])
    for idx, f_seg in enumerate(dense_frontier_segments):
        len_f_seg = f_seg.shape[0]
        f_seg_within = f_seg[within_dist[start_idx : start_idx + len_f_seg]] - offset
        start_idx += len_f_seg
        total_counts.append(len_f_seg)
        if f_seg_within.size == 0 or not is_visible[idx]:
            continue
        f_seg_masks[idx, f_seg_within[:, 0], f_seg_within[:, 1]] = 1

    # 6. Compute the amount of overlapping pixels between the left and right fogs
    # and each frontier segment

    # Reshape left_right_fogs to (2, N, H*W) and f_seg_masks to (S, H*W)
    left_right_fogs_flat = left_right_fogs.reshape(2, N, H * W)
    f_seg_masks_flat = f_seg_masks.reshape(S, H * W)
    # Shape: (2, N, H*W) x (S, H*W).T -> (2, N, S)
    overlap_count = np.matmul(left_right_fogs_flat, f_seg_masks_flat.T)
    overlap_percent = overlap_count.astype(np.float32) / np.array(total_counts)
    overlap_cumsum = np.cumsum(overlap_percent, axis=1)
    decimated = overlap_cumsum > 0.5
    # If decimated has no True values, return None
    if not decimated.any():
        return None

    # decimated[i, j, k]: i-th direction (left/right), j-th yaw, k-th frontier segment
    # Shape: (2, N, S) -> (2, N), i-th direction, j-th yaw (any frontier segment)
    decimated = decimated.any(axis=2)

    # 4. Decide action based on which direction has the first decimation
    first_true_left = decimated[0].argmax() if decimated[0].any() else np.inf
    first_true_right = decimated[1].argmax() if decimated[1].any() else np.inf

    return first_true_left < first_true_right
