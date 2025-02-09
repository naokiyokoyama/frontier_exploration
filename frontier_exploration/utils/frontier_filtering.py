from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.general_utils import wrap_heading


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
        self._fov = fov
        self._fov_length_px = fov_length_px
        self._image_bank: ImageBank = ImageBank(fov, fov_length_px)
        self._fseg_to_best_frontier_score: Dict[Tuple[int, ...], FrontierScore] = {}
        self.call_count: int = 0

    def reset(self) -> None:
        """Resets the filter's internal state, clearing all stored images and scores."""
        self._image_bank.reset()
        self._fseg_to_best_frontier_score = {}
        self.call_count = 0

    @CallCounter
    def score_and_filter_frontiers(
        self,
        curr_f_segments: List[np.ndarray],
        curr_cam_pos: Tuple[int, int],
        curr_cam_yaw: float,
        explored_area: np.ndarray,
        top_down_map: np.ndarray,
        curr_timestep_id: int,
        gt_idx: int = -1,
        filter: bool = True,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        assert (
            curr_timestep_id == self.call_count
        ), f"Expected timestep ID to be {self.call_count}, got {curr_timestep_id}"

        # 1. Regenerate a longer FOW for the current position and add it to the bank
        self._image_bank.add_fow(
            fow=reveal_fog_of_war(
                top_down_map=top_down_map,
                current_fog_of_war_mask=np.zeros_like(top_down_map),
                current_point=curr_cam_pos,
                current_angle=curr_cam_yaw,
                fov=self._fov,
                max_line_len=self._fov_length_px,
            ),
            camera_position=curr_cam_pos,
            camera_yaw=curr_cam_yaw,
            timestep_id=curr_timestep_id,
            explored_area=explored_area,
            obstacle_map=top_down_map,
        )

        if len(curr_f_segments) == 0:
            # No segments; just bail after adding FOW to bank
            return {}, {}

        # 2. For each of the current frontiers, identify the timestep ID and score of
        # the best image from the bank
        curr_f_tuples = [tuple(f.flatten()) for f in curr_f_segments]
        for f, f_tuple in zip(curr_f_segments, curr_f_tuples):
            # Update the best image and score for this frontier segment
            self._fseg_to_best_frontier_score[
                f_tuple
            ] = self._image_bank.get_best_frontier_score(f)

        if len(curr_f_segments) == 1:
            # Just one frontier; no filtering needed, bad_idx_to_good_idx is empty
            return {
                0: self._fseg_to_best_frontier_score[curr_f_tuples[0]].timestep_id
            }, {}

        # 3. Actual filtering occurs here; filter out frontiers that either have the
        # same or similar image as other frontiers with higher scores
        if filter:
            good_indices, bad_idx_to_good_idx = self._filter_frontiers_by_overlap(
                curr_f_tuples, gt_idx
            )
        else:
            # No filtering; all frontiers are good
            good_indices = list(range(len(curr_f_segments)))
            bad_idx_to_good_idx = {}

        # 4. At this point, the frontiers within good_indices are high-scorers and do
        # not appear similar to each other.
        good_indices_to_timestep = {
            idx: self._fseg_to_best_frontier_score[
                tuple(curr_f_segments[idx].flatten())
            ].timestep_id
            for idx in good_indices
        }

        return good_indices_to_timestep, bad_idx_to_good_idx

    def get_fog_of_war(self, timestep_id: int) -> np.ndarray:
        """
        Retrieve the field of view mask for a given timestep from the image bank.

        Args:
            timestep_id: The timestep identifier to lookup

        Returns:
            Binary mask representing the field of view at the given timestep
        """
        _, fow = self._image_bank.get_yaw_and_mask(timestep_id)
        return fow

    def _filter_frontiers_by_overlap(
        self, f_tuples: List[Tuple], gt_idx: int
    ) -> Tuple[List[int], Dict[int, int]]:
        all_finfos = []
        for f in f_tuples:
            f_score = self._fseg_to_best_frontier_score[f]
            camera_yaw, single_fog_of_war = self._image_bank.get_yaw_and_mask(
                f_score.timestep_id
            )

            all_finfos.append(
                FrontierInfo(
                    camera_yaw=camera_yaw,
                    single_fog_of_war=single_fog_of_war,
                    score=f_score.score,
                )
            )

        return filter_frontiers_by_overlap(all_finfos, gt_idx)


class ImageBank:
    """
    Stores and manages a collection of field-of-view masks and associated camera
    positions and yaws.

    Maintains a history of observations and provides methods to query and update the
    stored masks based on current exploration state.
    """

    def __init__(self, fov: float, fov_length_px: int) -> None:
        self.call_count: int = 0
        self._fov = fov
        self._fov_length_px = fov_length_px
        self._fow_masks: np.ndarray | None = None
        self._fow_masks_dilated: np.ndarray | None = None
        self._camera_positions: np.ndarray | None = None
        self._camera_yaws: np.ndarray | None = None
        self._mask_idx_to_timestep_id: np.ndarray | None = None
        self._num_masks_removed: int = 0

    def reset(self) -> None:
        """Resets the image bank to its initial state, clearing all stored data."""
        self.call_count = 0
        self._fow_masks = None
        self._fow_masks_dilated = None
        self._camera_positions = None
        self._camera_yaws = None
        self._mask_idx_to_timestep_id = None
        self._num_masks_removed = 0

    @property
    def fov(self) -> float:
        return self._fov

    @fov.setter
    def fov(self, radians: float) -> None:
        if not isinstance(radians, float):
            raise TypeError("FOV must be a float")
        if not 0 <= radians <= np.pi:
            raise ValueError("FOV must be between 0 and pi radians")
        if self.call_count != 0:
            raise RuntimeError("Cannot change FOV after adding images to the bank")
        self._fov = radians

    @CallCounter
    def add_fow(
        self,
        fow: np.ndarray,
        camera_position: np.ndarray,
        camera_yaw: float,
        timestep_id: int,
        explored_area: np.ndarray,
        obstacle_map: np.ndarray,
    ) -> None:
        assert (
            self.call_count == timestep_id
        ), f"{self.call_count = }, but {timestep_id = }"

        # fow must be dilated by 2 pixels to ensure overlap with frontiers that may
        # have been detected at the edge of the FOV
        fow_dilated = cv2.dilate(fow, np.ones((5, 5), np.uint8), iterations=1)

        # arr[None] unsqueezes arr; adds new axis at the beginning
        if self.call_count == 0:
            # First call since reset; initialize the bank
            self._fow_masks = fow[None]
            self._fow_masks_dilated = fow_dilated[None]
            self._camera_positions = camera_position[None]
            self._camera_yaws = np.array([camera_yaw])
            self._mask_idx_to_timestep_id = np.array([timestep_id])
        else:
            self._fow_masks = np.vstack([self._fow_masks, fow[None]])
            self._fow_masks_dilated = np.vstack(
                [self._fow_masks_dilated, fow_dilated[None]]
            )
            self._camera_positions = np.vstack(
                [self._camera_positions, camera_position[None]]
            )
            self._camera_yaws = np.hstack([self._camera_yaws, camera_yaw])
            self._mask_idx_to_timestep_id = np.hstack(
                [self._mask_idx_to_timestep_id, timestep_id]
            )

        self._refresh_bank(explored_area, obstacle_map)

        assert (
            self._fow_masks.shape[0]
            == self._fow_masks_dilated.shape[0]
            == self._camera_positions.shape[0]
            == self._camera_yaws.shape[0]
            == self._mask_idx_to_timestep_id.shape[0]
            == timestep_id + 1 - self._num_masks_removed
        ), (
            f"{self._fow_masks.shape = }\n"
            f"{self._fow_masks_dilated.shape = }\n"
            f"{self._camera_positions.shape = }\n"
            f"{self._camera_yaws.shape = }\n"
            f"{self._mask_idx_to_timestep_id.shape = }\n"
            f"{timestep_id + 1 - self._num_masks_removed = }"
        )

    def get_yaw_and_mask(self, timestep_id: int) -> Tuple[float, np.ndarray]:
        """
        Retrieves the camera yaw and FOV mask for a given timestep.

        Args:
           timestep_id: The timestep identifier to lookup

        Returns:
           Tuple containing:
               - Camera yaw angle in radians
               - Binary mask representing the field of view

        Raises:
           AssertionError: If timestep_id doesn't match exactly one stored mask
        """
        matching_inds = np.where(self._mask_idx_to_timestep_id == timestep_id)[0]
        assert len(matching_inds) == 1, (
            f"No matches or too many matches for '{timestep_id = }':\n"
            f"{matching_inds = }\n"
            f"{self._mask_idx_to_timestep_id = }"
        )
        mask_idx = matching_inds[0]

        return self._camera_yaws[mask_idx], self._fow_masks[mask_idx]

    def _refresh_bank(
        self, explored_area: np.ndarray, obstacle_map: np.ndarray
    ) -> None:
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
                fov=self._fov,
                max_line_len=self._fov_length_px,
            )
            self._fow_masks_dilated[idx] = cv2.dilate(
                self._fow_masks[idx], np.ones((5, 5), np.uint8), iterations=1
            )

        # 2. Identify masks that have at least some area outside the explored area and
        # remove the others
        # has_overlap = (self._fow_masks & ~explored_area).any(axis=(1, 2))
        # self._fow_masks = self._fow_masks[has_overlap]
        # self._fow_masks_dilated = self._fow_masks_dilated[has_overlap]
        # self._camera_positions = self._camera_positions[has_overlap]
        # self._camera_yaws = self._camera_yaws[has_overlap]
        # self._mask_idx_to_timestep_id = self._mask_idx_to_timestep_id[has_overlap]
        # self._num_masks_removed += (~has_overlap).sum()

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

        # Get masks that have any overlap with the frontier segment
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
        cam_pos = self._camera_positions[overlap_indices]  # shape: (M, 2)
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
        mask_idx = int(overlap_indices[np.argmax(mask_angle_sums)])
        best_score = np.max(mask_angle_sums)
        timestep_id = self._mask_idx_to_timestep_id[mask_idx]

        return FrontierScore(timestep_id=timestep_id, score=best_score)


@dataclass(frozen=True)
class FrontierInfo:
    """
    Contains information about the yaw, field-of-view, and best score associated with a
    frontier.

    Provides methods to compare frontiers for overlap based on camera yaw angles
    and field of view intersection.
    """

    camera_yaw: float
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


def interpolate_line(points: np.ndarray) -> np.ndarray:
    """
    Interpolate a sequence of 2D pixel locations to create a continuous line.
    Output has dtype int32, shape of (n, 2), and is in the same order as input.
    """
    if len(points) < 2:
        return points

    points = np.asarray(points)
    diffs = np.diff(points, axis=0)

    # Calculate required steps for each segment
    max_steps = np.max(np.abs(diffs), axis=1)
    steps = np.maximum(max_steps, 1)

    # Initialize list to store interpolated segments
    interpolated_points = []

    # Interpolate each segment
    for i in range(len(points) - 1):
        curr_steps = int(steps[i])
        segment_t = np.linspace(0, 1, curr_steps + 1).reshape(-1, 1)
        segment = points[i] + segment_t * diffs[i]
        interpolated_points.append(
            segment[:-1]
        )  # Exclude last point except for final segment

    # Add the last point
    interpolated_points.append(points[-1:])

    return np.vstack(interpolated_points).astype(np.int32)


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
