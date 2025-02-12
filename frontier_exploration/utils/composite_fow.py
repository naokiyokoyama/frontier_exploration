from typing import List

import numpy as np

from frontier_exploration.utils.fog_of_war import reveal_fog_of_war


class CompositeFOW:
    def __init__(self, *args, **kwargs):
        self._fov: List[float] = []
        self._fov_length_px: List[int] = []
        self._fow_masks: np.ndarray | None = None
        self._camera_positions: np.ndarray | None = None
        self._camera_yaws: np.ndarray | None = None
        self.__full_mask: np.ndarray | None = None

    def reset(self) -> None:
        self._fov = []
        self._fov_length_px = []
        self._fow_masks = None
        self._camera_positions = None
        self._camera_yaws = None
        self.__full_mask = None

    @property
    def full_mask(self) -> np.ndarray:
        if self.__full_mask is None:
            self.__full_mask = np.any(self._fow_masks, axis=0)
        return self.__full_mask

    def add_fow(
        self,
        fow: np.ndarray,
        camera_position: np.ndarray,
        camera_yaw: float,
        fov: float,
        fov_length_px: int,
        obstacle_map: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        # arr[None] unsqueezes arr; adds new axis at the beginning
        if self._fow_masks is None:
            # First call since reset; initialize the bank
            self._fow_masks = fow[None]
            self._camera_positions = camera_position[None]
            self._camera_yaws = np.array([camera_yaw], dtype=np.float16)
            self._fov = [fov]
            self._fov_length_px = [fov_length_px]
        else:
            self._fow_masks = np.vstack([self._fow_masks, fow[None]])
            self._camera_positions = np.vstack(
                [self._camera_positions, camera_position[None]]
            )
            self._camera_yaws = np.hstack([self._camera_yaws, camera_yaw])
            self._fov.append(fov)
            self._fov_length_px.append(fov_length_px)

        self._refresh_bank(obstacle_map)

        self.__full_mask = None

    def _refresh_bank(self, obstacle_map: np.ndarray) -> np.ndarray:
        """
        Update fogs that have any overlap with the current layout of obstacles, and
        remove fogs that are fully contained within the explored area
        """
        # 1. Identify masks that have any overlap with the current layout of obstacles
        # and update them.
        overlap_indices = np.nonzero((self._fow_masks & obstacle_map).any(axis=(1, 2)))[
            0
        ]
        for idx in overlap_indices:
            # Update the mask to consider the current layout of obstacles
            self._fow_masks[idx] = reveal_fog_of_war(
                top_down_map=obstacle_map,
                current_fog_of_war_mask=np.zeros_like(obstacle_map),
                current_point=self._camera_positions[idx],
                current_angle=self._camera_yaws[idx],
                fov=self._fov[idx],
                max_line_len=self._fov_length_px[idx],
            )

        return overlap_indices
