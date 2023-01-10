from typing import List, Optional

import cv2
import numpy as np
from numba import njit

from frontier_exploration.utils import closest_line_segment


def detect_frontier_waypoints(
    full_map: np.ndarray,
    explored_mask: np.ndarray,
    area_thresh: Optional[int] = -1,
    xy: Optional[np.ndarray] = None,
):
    frontiers = detect_frontiers(full_map, explored_mask, area_thresh)
    waypoints = frontier_waypoints(frontiers, xy)
    return waypoints


def detect_frontiers(
    full_map: np.ndarray,
    explored_mask: np.ndarray,
    area_thresh: Optional[int] = -1,
) -> List[np.ndarray]:
    """Detects frontiers in a map.

    Args:
        full_map (np.ndarray): White polygon on black image, where white is navigable.
        Mono-channel mask.
        explored_mask (np.ndarray): Portion of white polygon that has been seen already.
        This is also a mono-channel mask.
        area_thresh (int, optional): Minimum unexplored area (in pixels) needed adjacent
        to a frontier for that frontier to be valid. Defaults to -1.

    Returns:
        np.ndarray: A mono-channel mask where white contours represent each frontier.
    """
    # Find the contour of the explored area
    filtered_explored_mask = filter_out_small_unexplored(
        full_map, explored_mask, area_thresh
    )
    contours, _ = cv2.findContours(
        filtered_explored_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    unexplored_mask = np.where(filtered_explored_mask > 0, 0, full_map)
    unexplored_mask = cv2.blur(  # blurring for some leeway
        np.where(unexplored_mask > 0, 255, unexplored_mask), (3, 3)
    )
    frontiers = []
    # TODO: There shouldn't be more than one contour (only one explored area on map)
    for contour in contours:
        frontiers.extend(contour_to_frontiers(contour, unexplored_mask))
    return frontiers


def filter_out_small_unexplored(
    full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int
):
    """Edit the explored map to add small unexplored areas, which ignores their
    frontiers."""
    if area_thresh == -1:
        return explored_mask
    unexplored_mask = np.where(explored_mask > 0, 0, full_map)
    # Find contours in the unexplored mask
    contours, _ = cv2.findContours(
        unexplored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Add small unexplored areas to the explored map
    new_explored_mask = explored_mask.copy()
    for contour in contours:
        if cv2.contourArea(contour) < area_thresh:
            cv2.drawContours(new_explored_mask, [contour], -1, 255, -1)
    return new_explored_mask


@njit
def contour_to_frontiers(contour, unexplored_mask):
    """Given a contour from OpenCV, return a list of numpy arrays. Each array contains
    contiguous points forming a single frontier. The contour is assumed to be a set of
    contiguous points, but some of these points are not on any frontier, indicated by a
    value of 0 in the unexplored mask. This function will split the contour into
    multiple arrays"""
    bad_inds = []
    num_contours = len(contour)
    for idx in range(num_contours):
        point = contour[idx][0]
        x, y = point
        if unexplored_mask[y, x] == 0:
            bad_inds.append(idx)
    frontiers = np.split(contour, bad_inds)
    frontiers = [i for i in frontiers if len(i) > 1]
    # Combine the first and last frontier if adjacent (no bad points in between them)
    if not (0 in bad_inds or num_contours - 1 in bad_inds):
        last_frontier = frontiers.pop()
        frontiers[0] = np.concatenate((last_frontier, frontiers[0]))
    return frontiers


def frontier_waypoints(
    frontiers: List[np.ndarray], xy: Optional[np.ndarray] = None
) -> np.ndarray:
    """For each given frontier, returns the point on the frontier closest (euclidean
    distance) to the given coordinate. If coordinate is not given, will just return
    the midpoints of each frontier.

    Args:
        frontiers (List[np.ndarray]): list of arrays of shape (X, 1, 2), where each
        array is a frontier and X is NOT the same across arrays
        xy (np.ndarray): the given coordinate

    Returns:
        np.ndarray: array of waypoints, one for each frontier
    """
    if xy is None:
        return np.array([get_frontier_midpoint(i) for i in frontiers])
    return np.array([get_closest_frontier_point(xy, i) for i in frontiers])


@njit
def get_frontier_midpoint(frontier) -> np.ndarray:
    """Given a list of contiguous points (numpy arrays) representing a frontier, first
    calculate the total length of the frontier, then find the midpoint of the
    frontier"""
    # First, reshape and expand the frontier to be a 2D array of shape (X, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate((frontier[:-1], frontier[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # Calculate the length of each line segment
    line_lengths = np.sqrt(
        np.square(line_segments[:, 0, 0] - line_segments[:, 1, 0])
        + np.square(line_segments[:, 0, 1] - line_segments[:, 1, 1])
    )
    cum_sum = np.cumsum(line_lengths)
    total_length = cum_sum[-1]
    # Find the midpoint of the frontier
    midpoint = total_length / 2
    # Find the line segment that contains the midpoint
    line_segment_idx = np.argmax(cum_sum > midpoint)
    # Calculate the coordinates of the midpoint
    line_segment = line_segments[line_segment_idx]
    line_length = line_lengths[line_segment_idx]
    # Use the difference between the midpoint length and cumsum
    # to find the proportion of the line segment that the midpoint is at
    proportion = (midpoint - cum_sum[line_segment_idx - 1]) / line_length
    # Calculate the midpoint coordinates
    midpoint = line_segment[0] + proportion * (line_segment[1] - line_segment[0])
    return midpoint


def get_closest_frontier_point(xy, frontier):
    """Returns the point on the frontier closest to the given coordinate."""
    # First, reshape and expand the frontier to be a 2D array of shape (X, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate([frontier[:-1], frontier[1:]], axis=1).reshape(
        (-1, 2, 2)
    )
    closest_segment, closest_point = closest_line_segment(xy, line_segments)
    return closest_point


if __name__ == "__main__":
    import argparse
    import time

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--full_map", help="Path to full map image", default="full_map.png"
    )
    parser.add_argument(
        "-e",
        "--explored_mask",
        help="Path to explored map image",
        default="explored_mask.png",
    )
    parser.add_argument(
        "-a",
        "--area_thresh",
        help="Minimum unexplored area (in pixels) needed adjacent to a frontier for"
        "that frontier to be valid",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        help="Number of iterations to run the algorithm for timing purposes. Set to"
        "0 for no timing",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    # Read in the map
    full_map = cv2.imread(args.full_map, 0)
    # Read in the explored map
    explored_mask = cv2.imread(args.explored_mask, 0)
    times = []
    for _ in range(args.num_iterations + 1):
        start_time = time.time()
        waypoints = detect_frontier_waypoints(full_map, explored_mask, args.area_thresh)
        times.append(time.time() - start_time)
    if args.num_iterations > 0:
        # Skip first run as it's slower due to JIT compilation
        print(
            f"Avg. time taken for algorithm over {args.num_iterations} runs:",
            np.mean(times[1:]),
        )

    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.imshow(full_map, cmap="gray")
    plt.imshow(explored_mask, cmap="gray", alpha=0.5)
    for waypoint in waypoints:
        plt.scatter(waypoint[0], waypoint[1], c="red", s=50)
    plt.show()
