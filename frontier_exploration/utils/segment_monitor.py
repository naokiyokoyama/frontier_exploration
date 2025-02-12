from typing import List, Optional

import cv2
import numpy as np

from frontier_exploration.utils.fog_of_war import reveal_fog_of_war


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

    # 1. Compute the left and right fogs of war
    num_turns = int(np.pi * 2 / turn_angle)
    all_fogs = [
        cv2.erode(
            reveal_fog_of_war(
                top_down_map=obstacle_map,
                current_fog_of_war_mask=np.zeros_like(obstacle_map),
                current_point=camera_pos,
                current_angle=camera_yaw + i * turn_angle,
                fov=fov,
                max_line_len=max_line_len,
            ),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        for i in range(1, num_turns)
    ]

    H, W = obstacle_map.shape
    N = num_turns // 2
    left_right_fogs = np.empty((2, N, H, W), dtype=np.int32)
    for idx in range(N):
        left_right_fogs[0, idx] = all_fogs[idx]
        left_right_fogs[1, idx] = all_fogs[num_turns - 2 - idx]

    # 2. Create masks for each frontier segment
    S = len(frontier_segments)
    f_seg_masks = np.empty((S, H, W), dtype=np.int32)
    for idx, f_seg in enumerate(frontier_segments):
        f_seg_masks[idx, f_seg[:, 0], f_seg[:, 1]] = 1

    # 3. Compute the amount of overlapping pixels between the left and right fogs
    # and each frontier segment

    # Reshape left_right_fogs to (2, N, H*W) and f_seg_masks to (S, H*W)
    left_right_fogs_flat = left_right_fogs.reshape(2, N, H * W)
    f_seg_masks_flat = f_seg_masks.reshape(S, H * W)
    # Shape: (2, N, H*W) x (S, H*W).T -> (2, N, S)
    overlap_count = np.matmul(left_right_fogs_flat, f_seg_masks_flat.T)
    overlap_percent = overlap_count.astype(np.float32) / f_seg_masks.sum(axis=(1, 2))
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
