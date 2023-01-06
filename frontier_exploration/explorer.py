from typing import Optional

import cv2
import numpy as np


def detect_frontier_waypoints(
    full_map: np.ndarray, explored_map: np.ndarray, area_thresh: Optional[int] = -1
):
    mask = detect_frontiers(full_map, explored_map, area_thresh)
    return frontier_waypoints(mask)


def detect_frontiers(
    full_map: np.ndarray,
    explored_map: np.ndarray,
    area_thresh: Optional[int] = -1,
    whiten: Optional[bool] = True,
) -> np.ndarray:
    """Detects frontiers in a map.

    Args:
        full_map (np.ndarray): White polygon on black image, where white is navigable.
        Mono-channel mask.
        explored_map (np.ndarray): Portion of white polygon that has been seen already.
        This is also a mono-channel mask.
        area_thresh (int, optional): Minimum unexplored area (in pixels) needed adjacent
        to a frontier for that frontier to be valid. Defaults to -1.

    Returns:
        np.ndarray: A mono-channel mask where white contours represent each frontier.
    """
    if whiten:
        use_full_map, use_explored_map = [
            np.where(i > 0, 255, i) for i in (full_map, explored_map)
        ]
    else:
        use_full_map, use_explored_map = full_map, explored_map
    if area_thresh != -1:
        use_explored_map = filter_out_small_unexplored(
            use_full_map, use_explored_map, area_thresh
        )

    # Blur the full_map with a kernel of 3
    blurred_map = cv2.blur(use_explored_map, (3, 3))
    # Threshold the blurred map to get a binary image
    _, blurred_map = cv2.threshold(blurred_map, 0, 255, cv2.THRESH_BINARY)
    # Make a three channel image by stacking all images
    stacks = np.stack((use_full_map, use_explored_map, blurred_map), axis=2)
    # Make a single-channel mask where pixels are white if they are:
    # 1). navigable (white in full_map);
    # 2). unexplored (black in explored_map), but also;
    # 3). adjacent to explored pixels (white in blurred_map)
    mask = np.all(stacks == [255, 0, 255], axis=2).astype(np.uint8) * 255

    return mask


def filter_out_small_unexplored(
    full_map: np.ndarray, explored_map: np.ndarray, area_thresh: int
):
    """Edit the explored map to add small unexplored areas, which ignores their
    frontiers."""
    # Make mask of unexplored areas
    stacks = np.stack((full_map, explored_map), axis=2)
    unexplored_mask = np.all(stacks == [255, 0], axis=2).astype(np.uint8) * 255
    # Find contours in the unexplored mask
    contours, _ = cv2.findContours(
        unexplored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Add small unexplored areas to the explored map
    new_explored_map = explored_map.copy()
    for contour in contours:
        if cv2.contourArea(contour) < area_thresh:
            cv2.drawContours(new_explored_map, [contour], -1, 255, -1)
    return new_explored_map


def frontier_waypoints(frontier_mask: np.ndarray) -> np.ndarray:
    """Finds waypoints in a frontier mask.

    Args:
        frontier_mask (np.ndarray): A mono-channel mask where white contours represent
        each frontier.

    Returns:
        np.ndarray: An array of waypoints, where each waypoint is an array of (x, y)
    """
    # Find contours in the frontier mask
    contours, _ = cv2.findContours(
        frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return np.array([find_center_of_contour(contour) for contour in contours])


def find_center_of_contour(contour):
    moments = cv2.moments(contour)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    # Check if the center point is within the contour
    if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
        return center_x, center_y

    # Initialize variables to store the nearest point and minimum distance
    nearest_point = None
    min_distance = float("inf")

    for point in contour:
        # Find the distance between the center point and the current point
        distance = ((center_x - point[0][0]) ** 2) + ((center_y - point[0][1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = point[0]
    return nearest_point


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
        "--explored_map",
        help="Path to explored map image",
        default="explored_map.png",
    )
    parser.add_argument(
        "-a",
        "--area_thresh",
        help="Minimum unexplored area (in pixels) needed adjacent to a frontier for"
        "that frontier to be valid",
        type=float,
        default=-1,
    )
    args = parser.parse_args()

    # Read in the map
    full_map = cv2.imread(args.full_map, 0)
    # Read in the explored map
    explored_map = cv2.imread(args.explored_map, 0)
    start_time = time.time()
    # Detect the frontiers
    frontier_mask = detect_frontiers(full_map, explored_map, args.area_thresh)
    # Find the waypoints
    waypoints = frontier_waypoints(frontier_mask)
    print("Time taken for algorithm:", time.time() - start_time)

    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.imshow(full_map, cmap="gray")
    plt.imshow(explored_map, cmap="gray", alpha=0.5)
    plt.imshow(frontier_mask, cmap="gray", alpha=0.5)
    for waypoint in waypoints:
        plt.scatter(waypoint[0], waypoint[1], c="red", s=50)
    plt.show()
