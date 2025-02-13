from typing import List

import numpy as np

from frontier_exploration.utils.fog_of_war import reveal_fog_of_war


class CompositeFOWMixin:
    def __init__(self, *args, **kwargs):
        if hasattr(super(), "__init__"):
            super().__init__(*args, **kwargs)  # noqa

        self.full_mask: np.ndarray | None = None
        self._fov: List[float] = []
        self._fov_length_px: List[int] = []
        self._fow_masks: np.ndarray | None = None
        self._fow_positions: np.ndarray | None = None
        self._fow_yaws: np.ndarray | None = None
        self.__obstacle_map: np.ndarray | None = None
        self.__mask_bbox: np.ndarray | None = None

    def reset(self, *args, **kwargs) -> None:
        if hasattr(super(), "reset"):
            super().reset(*args, **kwargs)  # noqa
        self.full_mask = None
        self._fov = []
        self._fov_length_px = []
        self._fow_masks = None
        self._fow_positions = None
        self._fow_yaws = None
        self.__obstacle_map = None
        self.__mask_bbox = None

    def add_fow(
        self,
        fow: np.ndarray,
        fow_position: np.ndarray,
        fow_yaw: float,
        fov: float,
        fov_length_px: int,
        obstacle_map: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        fow_bool = fow.astype(bool)
        bbox = get_mask_bbox(fow_bool)
        # arr[None] unsqueezes arr; adds new axis at the beginning
        if self._fow_masks is None:
            # First call since reset; initialize the bank
            self._fow_masks = fow_bool[None]
            self._fow_positions = fow_position[None].astype(np.float16)
            self._fow_yaws = np.array([fow_yaw], dtype=np.float16)
            self._fov = [fov]
            self._fov_length_px = [fov_length_px]
            self.__mask_bbox = bbox[None]
            self.full_mask = fow_bool.copy()
        else:
            self._fow_masks = np.vstack([self._fow_masks, fow_bool[None]])
            self._fow_positions = np.vstack([self._fow_positions, fow_position[None]])
            self._fow_yaws = np.hstack([self._fow_yaws, fow_yaw])
            self._fov.append(fov)
            self._fov_length_px.append(fov_length_px)
            self.__mask_bbox = np.vstack([self.__mask_bbox, bbox[None]])

        overlap_indices = self._refresh_bank(obstacle_map, bbox)
        if overlap_indices.size == 0:
            x_min, y_min, x_max, y_max = bbox + np.array([-1, -1, 1, 1])
            self.full_mask[y_min:y_max, x_min:x_max] |= fow_bool[
                y_min:y_max, x_min:x_max
            ]

    def _refresh_bank(self, obstacle_map: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Update fogs that have any overlap with the current layout of obstacles, and
        remove fogs that are fully contained within the explored area
        """
        # 1. Identify masks that have any overlap with the updated portion of the
        # current layout of obstacles, and the region that the overlap occurs.
        obstacle_map_bool = obstacle_map.astype(bool)
        if self.__obstacle_map is not None:
            diff = np.logical_xor(obstacle_map_bool, self.__obstacle_map)
            if not diff.any():
                return np.array([])
            x_min, y_min, x_max, y_max = get_mask_bbox(diff) + np.array([-1, -1, 1, 1])
        else:
            x_min, y_min, x_max, y_max = (
                0,
                0,
                obstacle_map.shape[1],
                obstacle_map.shape[0],
            )
        self.__obstacle_map = obstacle_map_bool

        candidate_mask = check_box_intersections(self.__mask_bbox, bbox)
        if not candidate_mask.any():
            return np.array([])  # No masks intersect with the updated area

        candidate_indices = np.nonzero(candidate_mask)[0]
        candidates = self._fow_masks[candidate_indices]
        overlap = (
            candidates[:, y_min:y_max, x_min:x_max]
            & obstacle_map[y_min:y_max, x_min:x_max]
        )
        overlap_indices = np.nonzero(overlap.any(axis=(1, 2)))[0]
        overlap_indices = candidate_indices[overlap_indices]
        if overlap_indices.size == 0:
            return np.array([])  # No intersecting masks have any overlap

        # Record the fully affected area to update later for full_mask
        x_min, y_min = np.min(self.__mask_bbox[overlap_indices], axis=0)[:2] - 1
        x_max, y_max = np.max(self.__mask_bbox[overlap_indices], axis=0)[2:] + 1

        for idx in overlap_indices:
            # Update the mask to consider the current layout of obstacles
            self._fow_masks[idx] = reveal_fog_of_war(
                top_down_map=obstacle_map,
                current_fog_of_war_mask=np.zeros_like(obstacle_map),
                current_point=self._fow_positions[idx],
                current_angle=self._fow_yaws[idx],
                fov=self._fov[idx],
                max_line_len=self._fov_length_px[idx],
            ).astype(bool)
            self.__mask_bbox[idx] = get_mask_bbox(self._fow_masks[idx])

        # 2. Update the full mask
        overlap_mask = check_box_intersections(
            self.__mask_bbox, np.array([x_min, y_min, x_max, y_max])
        )

        self.full_mask[y_min:y_max, x_min:x_max] = np.any(
            self._fow_masks[overlap_mask, y_min:y_max, x_min:x_max],
            axis=0,
        )

        return overlap_indices


def get_mask_bbox(binary_mask: np.ndarray) -> np.ndarray:
    """
    Find the bounding box coordinates for True/1 values in a binary mask using numpy.

    Args:
        binary_mask: 2D numpy array of boolean or 0/1 values

    Returns:
        np.ndarray: (x_min, y_min, x_max, y_max) representing the bounding box
                    coordinates
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not np.any(rows) or not np.any(cols):  # No True values found
        return np.array([-1, -1, -1, -1], dtype=int)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return np.array([x_min, y_min, x_max, y_max], dtype=int)


def check_box_intersections(boxes: np.ndarray, query_box: np.ndarray) -> np.ndarray:
    """
    Check which boxes intersect with a query box.

    Args:
        boxes: Array of shape (N, 4) containing N boxes with coordinates
               (xmin, ymin, xmax, ymax)
        query_box: Array of shape (4,) containing a single box coordinates
                   (xmin, ymin, xmax, ymax)

    Returns:
        Boolean array of shape (N,) where True indicates intersection with query_box
    """
    # Compute the intersection coordinates
    intersect_xmin = np.maximum(boxes[:, 0], query_box[0])
    intersect_ymin = np.maximum(boxes[:, 1], query_box[1])
    intersect_xmax = np.minimum(boxes[:, 2], query_box[2])
    intersect_ymax = np.minimum(boxes[:, 3], query_box[3])

    # Check if there is a valid intersection
    # (both width and height of intersection must be positive)
    return np.bitwise_and(
        intersect_xmax > intersect_xmin, intersect_ymax > intersect_ymin
    )
