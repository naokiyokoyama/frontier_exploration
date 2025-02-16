from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from frontier_exploration.utils.composite_fow import CompositeFOWMixin
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.general_utils import (
    calculate_perpendicularity,
    interpolate_line,
    wrap_heading,
)


class CallCounter:
    """
    A descriptor that counts the number of times a method is called on an instance.

    When applied to a method, increments a call_count attribute on the instance each
    time the method is called.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        # Bind the method to the instance
        def wrapped(*args, **kwargs):
            ret = self.func(obj, *args, **kwargs)  # Call original method
            obj.call_count += 1  # Increment the counter
            return ret

        return wrapped


@dataclass(frozen=True)
class FrontierScore:
    timestep_id: int
    score: float


@dataclass(frozen=True)
class FrontierTimestepData:
    good_indices_to_timestep: Dict[int, int] = field(default_factory=dict)
    bad_idx_to_good_idx: Dict[int, int] = field(default_factory=dict)


@dataclass
class FrontierFilterData:
    filtered: FrontierTimestepData = FrontierTimestepData()
    unfiltered: FrontierTimestepData = FrontierTimestepData()
    unscored_filtered: FrontierTimestepData = FrontierTimestepData()
    unscored_unfiltered: FrontierTimestepData = FrontierTimestepData()

    def set_single_frontier(self, timestep: int, unscored_timestep: int = -1) -> None:
        self.filtered = FrontierTimestepData(good_indices_to_timestep={0: timestep})
        self.unfiltered = FrontierTimestepData(good_indices_to_timestep={0: timestep})
        if unscored_timestep != -1:
            self.unscored_filtered = FrontierTimestepData(
                good_indices_to_timestep={0: unscored_timestep}
            )
            self.unscored_unfiltered = FrontierTimestepData(
                good_indices_to_timestep={0: unscored_timestep}
            )

    def verify(self) -> None:
        if self.unfiltered.good_indices_to_timestep:
            # Filtered should be a subset of unfiltered
            assert all(
                self.unfiltered.good_indices_to_timestep[k] == v
                for k, v in self.filtered.good_indices_to_timestep.items()
            )
        if self.unscored_unfiltered.good_indices_to_timestep:
            # Unscored filtered should be a subset of unscored unfiltered
            assert all(
                self.unscored_unfiltered.good_indices_to_timestep[k] == v
                for k, v in self.unscored_filtered.good_indices_to_timestep.items()
            )
        for d in (
            self.filtered,
            self.unfiltered,
            self.unscored_filtered,
            self.unscored_unfiltered,
        ):
            # Ensure that the timesteps are all unique
            assert len(set(d.good_indices_to_timestep.values())) == len(
                d.good_indices_to_timestep
            )


class FrontierFilter:
    """
    Filters frontier segments based on field of view (FOV) and overlap criteria.

    Maintains a history of previous observations and scores frontier segments based
    on their visibility from different camera positions.

    Args:
        fov: Field of view angle in radians
        fov_length_px: Maximum length of field of view in pixels
    """

    def __init__(self, fov: float, fov_length_px: int) -> None:
        self.__fov = fov
        self.__fov_length_px = fov_length_px
        self._f_scorer: FrontierScorer = FrontierScorer(fov, fov_length_px)
        self._fseg_to_best_frontier_score: Dict[Tuple[int, ...], FrontierScore] = {}
        self._fseg_to_most_recent_timestep: Dict[Tuple[int, ...], int] = {}
        self._camera_pose_to_timestep: Dict[Tuple[int, int, float], int] = {}
        self.call_count: int = 0

    def reset(self) -> None:
        """Resets the filter's internal state, clearing all stored images and scores."""
        self._f_scorer.reset()
        self._fseg_to_best_frontier_score = {}
        self._fseg_to_most_recent_timestep = {}
        self._camera_pose_to_timestep = {}
        self.call_count = 0

    @CallCounter
    def score_and_filter_frontiers(
        self,
        curr_f_segments: List[np.ndarray],
        curr_cam_pos: Tuple[int, int],
        curr_cam_yaw: float,
        top_down_map: np.ndarray,
        curr_timestep_id: int,
        gt_idx: int = -1,
        filter: bool = True,
        return_all: bool = False,
    ) -> FrontierFilterData:
        assert (
            curr_timestep_id == self.call_count
        ), f"Expected timestep ID to be {self.call_count}, got {curr_timestep_id}"

        # 1. Regenerate a longer FOW for the current position and add it to the bank
        fow = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=np.zeros_like(top_down_map),
            current_point=curr_cam_pos,
            current_angle=curr_cam_yaw,
            fov=self.__fov,
            max_line_len=self.__fov_length_px,
        )

        self._f_scorer.add_fow(
            fow=fow,
            fow_position=curr_cam_pos,
            fow_yaw=curr_cam_yaw,
            obstacle_map=top_down_map,
            timestep_id=curr_timestep_id,
        )

        if len(curr_f_segments) == 0:
            # No segments; just bail after adding FOW to bank
            return result

        # 2. For each of the current frontiers, identify the timestep ID and score of
        # the best image from the bank
        curr_f_tuples = [tuple(f.flatten()) for f in curr_f_segments]
        for f, f_tuple in zip(curr_f_segments, curr_f_tuples):
            # Update the best image and score for this frontier segment
            try:
                self._fseg_to_best_frontier_score[
                    f_tuple
                ] = self._f_scorer.get_best_frontier_score(f)
            except IndexError:
                self._fseg_to_best_frontier_score[f_tuple] = FrontierScore(
                    timestep_id=self.call_count, score=-1
                )

        if return_all:
            # Determine the most recent timestep for each frontier
            curr_pose = (
                int(curr_cam_pos[0]),
                int(curr_cam_pos[1]),
                float(curr_cam_yaw),
            )
            if curr_pose not in self._camera_pose_to_timestep:
                self._camera_pose_to_timestep[curr_pose] = curr_timestep_id
            for f in curr_f_tuples:
                if f not in self._fseg_to_most_recent_timestep:
                    # New frontier updated by current timestep; update the most
                    # recent timestep
                    self._fseg_to_most_recent_timestep[
                        f
                    ] = self._camera_pose_to_timestep[curr_pose]

        if len(curr_f_segments) == 1:
            # Just one frontier; no filtering needed, bad_idx_to_good_idx is empty
            unscored_timestep = self._fseg_to_most_recent_timestep.get(
                curr_f_tuples[0], -1
            )
            result.set_single_frontier(curr_timestep_id, unscored_timestep)
            return result

        # 3. Actual filtering occurs here; filter out frontiers that either have the
        # same or similar image as other frontiers with higher scores
        if return_all or not filter:
            # Don't filter by overlap; however, should still filter out frontiers that
            # map to the same timestep as another frontier with a higher score

            # Create [(idx, f_score), ...] in descending order of f_score.score
            sorted_idx_and_f_scores: List[Tuple[int, FrontierScore]] = []
            timestep_to_best_idx: Dict[int, int] = {}
            for idx, f in enumerate(curr_f_tuples):
                if idx == gt_idx:
                    timestep_to_best_idx[
                        self._fseg_to_best_frontier_score[f].timestep_id
                    ] = idx
                else:
                    sorted_idx_and_f_scores.append(
                        (idx, self._fseg_to_best_frontier_score[f])
                    )
            sorted_idx_and_f_scores = sorted(
                sorted_idx_and_f_scores, key=lambda x: x[1].score, reverse=True
            )
            bad_idx_to_good_idx: Dict[int, int] = {}
            for idx, f_score in sorted_idx_and_f_scores:
                # Add current index to timestep_to_best_idx if none of the indices added
                # so far share the same timestep as the current index
                if f_score.timestep_id not in timestep_to_best_idx:
                    timestep_to_best_idx[f_score.timestep_id] = idx
                else:
                    # Map the bad index to the good one
                    bad_idx_to_good_idx[idx] = timestep_to_best_idx[f_score.timestep_id]

            good_indices = list(timestep_to_best_idx.values())
            good_indices_to_timestep = {
                idx: self._fseg_to_best_frontier_score[curr_f_tuples[idx]].timestep_id
                for idx in good_indices
            }
            result.unfiltered = FrontierTimestepData(
                good_indices_to_timestep=good_indices_to_timestep,
                bad_idx_to_good_idx=bad_idx_to_good_idx,
            )

        if return_all:
            # Unscored and unfiltered
            timestep_to_indices = {
                self._fseg_to_most_recent_timestep[curr_f_tuples[gt_idx]]: gt_idx
            }
            bad_idx_to_good_idx = {}
            for idx, f in enumerate(curr_f_tuples):
                t_step = self._fseg_to_most_recent_timestep[f]
                if t_step not in timestep_to_indices:
                    timestep_to_indices[t_step] = idx
                else:
                    bad_idx_to_good_idx[idx] = timestep_to_indices[t_step]
            good_indices = list(timestep_to_indices.values())
            good_indices_to_timestep = {
                idx: self._fseg_to_most_recent_timestep[curr_f_tuples[idx]]
                for idx in good_indices
            }
            result.unscored_unfiltered = FrontierTimestepData(
                good_indices_to_timestep=good_indices_to_timestep,
                bad_idx_to_good_idx=bad_idx_to_good_idx,
            )

        if return_all or filter:
            good_indices, bad_idx_to_good_idx = self._filter_frontiers_by_overlap(
                curr_f_tuples, gt_idx
            )
            good_indices_to_timestep = {
                idx: self._fseg_to_best_frontier_score[curr_f_tuples[idx]].timestep_id
                for idx in good_indices
            }
            result.filtered = FrontierTimestepData(
                good_indices_to_timestep=good_indices_to_timestep,
                bad_idx_to_good_idx=bad_idx_to_good_idx,
            )

        if return_all:
            all_finfos = []
            for idx, f in enumerate(curr_f_tuples):
                t_step = self._fseg_to_most_recent_timestep[f]
                yaw, mask = self._f_scorer.get_yaw_and_mask(t_step)
                seg_ends = curr_f_segments[idx][[0, -1]].reshape(1, 2, 2)
                score = 1 - calculate_perpendicularity(
                    self._f_scorer.get_cam_pos(t_step), seg_ends
                )
                all_finfos.append(
                    FrontierInfo(
                        fow_yaw=yaw, single_fog_of_war=mask, score=score.item()
                    )
                )

            good_indices, bad_idx_to_good_idx = filter_frontiers_by_overlap(
                all_finfos, gt_idx
            )
            good_indices_to_timestep = {
                idx: self._fseg_to_most_recent_timestep[curr_f_tuples[idx]]
                for idx in good_indices
            }
            result.unscored_filtered = FrontierTimestepData(
                good_indices_to_timestep=good_indices_to_timestep,
                bad_idx_to_good_idx=bad_idx_to_good_idx,
            )

        result.verify()

        return result

    def get_fog_of_war(self, timestep_id: int) -> np.ndarray:
        """
        Retrieve the field of view mask for a given timestep from the image bank.

        Args:
            timestep_id: The timestep identifier to lookup

        Returns:
            Binary mask representing the field of view at the given timestep
        """
        _, fow = self._f_scorer.get_yaw_and_mask(timestep_id)
        return fow

    def _filter_frontiers_by_overlap(
        self, f_tuples: List[Tuple], gt_idx: int
    ) -> Tuple[List[int], Dict[int, int]]:
        all_finfos = []
        for f in f_tuples:
            f_score = self._fseg_to_best_frontier_score[f]
            fow_yaw, single_fog_of_war = self._f_scorer.get_yaw_and_mask(
                f_score.timestep_id
            )

            all_finfos.append(
                FrontierInfo(
                    fow_yaw=fow_yaw,
                    single_fog_of_war=single_fog_of_war,
                    score=f_score.score,
                )
            )

        return filter_frontiers_by_overlap(all_finfos, gt_idx)


class FrontierScorer(CompositeFOWMixin):
    """
    Stores and manages a collection of field-of-view masks and associated camera
    positions and yaws.

    Maintains a history of observations and provides methods to query and update the
    stored masks based on current exploration state.
    """

    def __init__(self, fov: float, fov_length_px: int) -> None:
        super().__init__()
        self.__fov = fov
        self.__fov_length_px = fov_length_px

        self.call_count: int = 0
        self._fow_masks_dilated: np.ndarray | None = None

    def reset(self) -> None:
        """Resets the image bank to its initial state, clearing all stored data."""
        super().reset()
        self.call_count = 0
        self._fow_masks_dilated = None

    @property
    def fov(self) -> float:
        return self.__fov

    @fov.setter
    def fov(self, radians: float) -> None:
        if not isinstance(radians, float):
            raise TypeError("FOV must be a float")
        if not 0 <= radians <= np.pi:
            raise ValueError("FOV must be between 0 and pi radians")
        if self.call_count != 0:
            raise RuntimeError("Cannot change FOV after adding images to the bank")
        self.__fov = radians

    @CallCounter
    def add_fow(
        self,
        fow: np.ndarray,
        fow_position: np.ndarray,
        fow_yaw: float,
        obstacle_map: np.ndarray,
        timestep_id: int,
    ) -> None:
        assert (
            self.call_count == timestep_id
        ), f"{self.call_count = }, but {timestep_id = }"

        # fow must be dilated by 2 pixels to ensure overlap with frontiers that may
        # have been detected at the edge of the FOV
        fow_dilated = cv2.dilate(
            fow.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
        ).astype(bool)
        if self._fow_masks_dilated is None:
            self._fow_masks_dilated = fow_dilated[None]
        else:
            self._fow_masks_dilated = np.vstack(
                [self._fow_masks_dilated, fow_dilated[None]]
            )

        super().add_fow(
            fow=fow,
            fow_position=fow_position,
            fow_yaw=fow_yaw,
            fov=self.__fov,
            fov_length_px=self.__fov_length_px,
            obstacle_map=obstacle_map,
        )

        assert (
            self._fow_masks.shape[0]
            == self._fow_masks_dilated.shape[0]
            == self._fow_positions.shape[0]
            == self._fow_yaws.shape[0]
            == timestep_id + 1
        ), (
            f"{self._fow_masks.shape = }\n"
            f"{self._fow_masks_dilated.shape = }\n"
            f"{self._fow_positions.shape = }\n"
            f"{self._fow_yaws.shape = }\n"
            f"{timestep_id + 1 = }"
        )

    def _refresh_bank(self, obstacle_map: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        overlap_indices = super()._refresh_bank(obstacle_map, bbox)
        for idx in overlap_indices:
            # Update the mask to consider the current layout of obstacles
            self._fow_masks_dilated[idx] = cv2.dilate(
                self._fow_masks[idx].astype(np.uint8),
                np.ones((5, 5), np.uint8),
                iterations=1,
            ).astype(bool)

        return overlap_indices

    def get_yaw_and_mask(self, timestep_id: int) -> Tuple[float, np.ndarray]:
        """
        Retrieves the camera yaw and FOV mask for a given timestep.

        Args:
           timestep_id: The timestep identifier to lookup

        Returns:
           Tuple containing:
               - Camera yaw angle in radians
               - Binary mask representing the field of view
        """
        return self._fow_yaws[timestep_id], self._fow_masks[timestep_id]

    def get_cam_pos(self, timestep_id: int) -> np.ndarray:
        """
        Retrieves the camera position for a given timestep.

        Args:
           timestep_id: The timestep identifier to lookup

        Returns:
           Camera position as a numpy array
        """
        return self._fow_positions[timestep_id]

    def get_best_frontier_score(self, frontier_segment: np.ndarray) -> FrontierScore:
        """
        Calculates the best score for a frontier segment based on stored masks.

        Identifies masks that overlap with the frontier segment and scores them based
        on viewing angles. Returns the timestep ID and score of the best-scoring mask.

        Args:
            frontier_segment: Array of points representing a frontier segment

        Returns:
            FrontierScore containing the timestep ID and score of the best mask
        """
        # 1. Identify masks in the bank that have any overlap with the frontier segment

        # Interpolate the frontier segment to get a dense set of points
        fs_dense = interpolate_line(frontier_segment)

        # Convert points to integer indices
        y_coords = fs_dense[:, 0].astype(int)
        x_coords = fs_dense[:, 1].astype(int)

        # Identify masks that have any overlap with the frontier segment. The array
        # overlap_coords has shape (M, L), where M is the number of masks in the bank
        # and L is the number of points in the frontier segment. overlap_coords[i, j] is
        # True if in the i-th mask, the j-th point in the frontier segment is set
        overlap_coords = self._fow_masks_dilated[:, y_coords, x_coords].copy()

        # Generate a list of indices of masks that have any overlap with the frontier
        overlap_indices = np.nonzero(np.any(overlap_coords, axis=1))[0]

        # 2. Calculate the score for each mask that has overlap with the frontier
        # segment. The score is equal to the sum of angles between the mask's camera
        # position and the first and last points of each subsegment that overlaps with
        # the mask.

        # Remove rows that have no overlap with the frontier segment
        overlap_coords = overlap_coords[overlap_indices]

        # Filter out subsegments that have a length of 1
        overlap_coords = process_adjacent_columns(overlap_coords)
        overlap_indices = overlap_indices[overlap_coords.sum(axis=1) > 1]
        overlap_coords = overlap_coords[overlap_coords.sum(axis=1) > 1]

        # Get the start and end indices of each subsegment for each mask;
        # both masks have shape (M, L)
        left_ends_mask, right_ends_mask = get_span_boundaries(overlap_coords)

        # Calculate the number of subsegments for each mask. num_subsegments has shape
        # (M,) where M is the number of masks, and the sum of all elements is the total
        # number of subsegments across all masks, S, where S >= M.
        num_subsegments = np.sum(left_ends_mask, axis=1)

        # Get the start and end points of all subsegments at once
        # fs_dense_broadcast has shape (M, L, 2), where N is the number of masks, L is
        # the number of points in the frontier segment, and 2 is the x and y
        fs_dense_broadcast = np.broadcast_to(
            fs_dense, (overlap_coords.shape[0], *fs_dense.shape)
        )
        # start_points and end_points have shape (S, 2), where S is the total number of
        # subsegments across all masks, and 2 is the x and y. S >= M.
        start_points = fs_dense_broadcast[left_ends_mask]
        end_points = fs_dense_broadcast[right_ends_mask]
        # Calculate the angles between the camera position and the start and end points
        # of each subsegment. overlapping_subsegments has shape (S, 2, 2).
        overlapping_subsegments = np.stack((start_points, end_points), axis=1)
        cam_pos = self._fow_positions[overlap_indices]  # shape: (M, 2)
        # cam_pos_per_subsegment has shape (S, 2)
        cam_pos_per_subsegment = np.repeat(cam_pos, num_subsegments, axis=0)
        all_angles = calculate_angles(
            # shapes: (S, 2, 2), (S, 2) -> (S,)
            overlapping_subsegments,
            cam_pos_per_subsegment,
        )

        # Calculate the score for each mask
        mask_angle_sums = group_sums(all_angles, num_subsegments)  # shape: (N,)

        # 3. Return the timestep of the mask with the highest score and the score itself
        timestep_id = int(overlap_indices[np.argmax(mask_angle_sums)])
        best_score = np.max(mask_angle_sums)

        return FrontierScore(timestep_id=timestep_id, score=best_score)


@dataclass(frozen=True)
class FrontierInfo:
    """
    Contains information about the yaw, field-of-view, and best score associated with a
    frontier.

    Provides methods to compare frontiers for overlap based on camera yaw angles
    and field of view intersection.
    """

    fow_yaw: float
    single_fog_of_war: np.ndarray
    score: float

    def is_overlapping(self, other: "FrontierInfo") -> bool:
        """
        Two frontiers overlap if the difference between their two camera yaws is less
        than 45 degrees and the overlap between their single fog of wars is greater
        than 75%.
        """
        assert self.single_fog_of_war.shape == other.single_fog_of_war.shape

        if np.array_equal(self.single_fog_of_war, other.single_fog_of_war):
            return True

        yaw_diff = abs(wrap_heading(self.fow_yaw - other.fow_yaw))
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
            percentage_overlap = min(overlap / nonzero1, overlap / nonzero2)
        return percentage_overlap > 0.75


def filter_frontiers_by_overlap(
    all_finfos: List[FrontierInfo], gt_idx: int
) -> Tuple[List[int], Dict[int, int]]:
    indices_to_keep: List[int] = []
    bad_idx_to_good_idx: Dict[int, int] = {}

    for idx, finfo in enumerate(all_finfos):
        # For each frontier, check if any frontier so far in the filtered list overlaps
        # with it. If not, add it to the filtered list. If it does, for all overlapping
        # frontiers, keep the highest scorer, and, importantly, remove the other
        # overlapping frontiers from filtered list and indices to keep.
        is_overlapping = {
            i: finfo.is_overlapping(all_finfos[i]) for i in indices_to_keep
        }

        if not any(is_overlapping.values()):
            # No overlaps -> no filtering needed
            indices_to_keep.append(idx)
            continue

        # Current frontier overlaps with at least one frontier in the filtered list;
        # remove these overlapping frontiers from the filtered list
        overlapping_indices = [i for i, v in is_overlapping.items() if v]
        indices_to_keep = [i for i in indices_to_keep if i not in overlapping_indices]

        # Add current frontier to the overlapping ones
        overlapping_indices.append(idx)

        # Keep the best frontier among the overlapping ones
        if gt_idx in overlapping_indices:
            # The ground truth is one of the overlapping ones; keep it
            best_idx = gt_idx
        else:
            best_idx = max(overlapping_indices, key=lambda i: all_finfos[i].score)

        # Add the best frontier to the filtered list
        indices_to_keep.append(best_idx)

        # Map the bad indices to the good ones
        for i in [_ for _ in overlapping_indices if _ != best_idx]:
            bad_idx_to_good_idx[i] = best_idx

    return sorted(indices_to_keep), bad_idx_to_good_idx


def process_adjacent_columns(arr: np.ndarray) -> np.ndarray:
    """
    Process a binary 2D array to create a new array where elements are True only if both
    the original element is True and at least one adjacent element in the same row is True.

    Args:
        arr (np.ndarray): Input binary array of shape (N, L) where L >= 2

    Returns:
        np.ndarray: Processed binary array of the same shape as input

    Raises:
        AssertionError: If L < 2
    """
    assert arr.shape[1] >= 2, f"Second dimension must be at least 2, got {arr.shape}"

    # Create shifted versions of the array for checking adjacent columns
    left_shift = np.pad(
        arr[:, 1:], ((0, 0), (0, 1)), mode="constant", constant_values=False
    )
    right_shift = np.pad(
        arr[:, :-1], ((0, 0), (1, 0)), mode="constant", constant_values=False
    )

    # Element is True if original is True and either left or right neighbor is True
    return arr & (left_shift | right_shift)


def get_span_boundaries(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binary masks indicating the start and end positions of True sequences in a
    binary mask.

    Args:
        mask: Binary numpy array of shape (N, L) where N is the batch size and L is the
             sequence length. Values should be boolean or 0/1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two binary arrays of same shape as input (N, L):
            - left_ends: True where mask[i,j] is True and (j=0 or mask[i,j-1] is False)
            - right_ends: True where mask[i,j] is True and (j=L-1 or mask[i,j+1] is
                         False)
    """
    # Ensure input is boolean
    mask = mask.astype(bool)
    N, L = mask.shape

    # Create shifted versions of the mask for comparison
    pad_left = np.zeros((N, 1), dtype=bool)
    pad_right = np.zeros((N, 1), dtype=bool)

    mask_shifted_right = np.hstack([pad_left, mask[:, :-1]])
    mask_shifted_left = np.hstack([mask[:, 1:], pad_right])

    left_ends_mask = mask & ~mask_shifted_right
    right_ends_mask = mask & ~mask_shifted_left

    return left_ends_mask, right_ends_mask


def calculate_angles(
    point_pairs: np.ndarray, reference_points: np.ndarray
) -> np.ndarray:
    """
    Calculate angles between lines formed by connecting reference points to pairs of
    points.

    Args:
        point_pairs: Array of shape (N, 2, 2) where each element contains two 2D
                    coordinates
        reference_points: Array of shape (N, 2) containing N 2D coordinates

    Returns:
        Array of shape (N,) containing angles in radians between the lines formed by
        connecting each reference point to its corresponding pair of points

    Example:
        >>> pairs = np.array([[[0, 0], [1, 0]], [[2, 2], [3, 3]]])  # Two pairs of pts
        >>> refs = np.array([[0, 1], [1, 1]])  # Two reference points
        >>> angles = calculate_angles(pairs, refs)  # shape: (2,)
    """
    # Validate input shapes
    assert point_pairs.shape == (reference_points.shape[0], 2, 2), (
        f"Expected shapes of (N, 2, 2) and (N, 2), got {point_pairs.shape} and "
        f"{reference_points.shape}"
    )

    # Calculate vectors from reference points to each point in the pairs
    vectors = point_pairs - reference_points[:, np.newaxis, :]

    # Calculate dot products between vectors
    dot_products = np.sum(vectors[:, 0] * vectors[:, 1], axis=1)

    # Calculate magnitudes of vectors
    magnitudes = np.sqrt(np.sum(vectors**2, axis=2))

    # Calculate angles using arccos of dot product divided by product of magnitudes
    angles = np.arccos(dot_products / (magnitudes[:, 0] * magnitudes[:, 1] + 1e-6))

    return angles


def group_sums(values: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
    """
    Compute sums of consecutive groups of elements from an array, where group sizes are
    specified.

    Args:
        values: Float array of shape (N,) containing values to be summed
        group_sizes: Integer array of shape (M,) specifying the size of each group

    Returns:
        Float array of shape (M,) containing sums of consecutive groups

    Raises:
        AssertionError: If any group size is not positive or if sum of group sizes
                       doesn't match length of values array

    Example:
        >>> values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> group_sizes = np.array([2, 3])
        >>> group_sums(values, group_sizes)
        array([3., 12.])  # [1.0+2.0, 3.0+4.0+5.0]
    """
    # Input validation
    assert np.all(group_sizes > 0), "All group sizes must be positive"
    assert np.sum(group_sizes) == len(
        values
    ), "Sum of group sizes must equal length of values array"

    # Calculate the indices where each group ends
    end_indices = np.cumsum(group_sizes)
    # Calculate the indices where each group starts
    start_indices = np.roll(end_indices, 1)
    start_indices[0] = 0

    # Use numpy's add.reduceat to efficiently compute sums for each group
    return np.add.reduceat(values, start_indices)
