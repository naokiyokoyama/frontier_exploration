from typing import List, Optional

import cv2
import numpy as np


def detect_frontiers(
    full_map: np.ndarray, explored_map: np.ndarray, area_thresh: Optional[int] = -1
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
    if area_thresh != -1:
        filter_out_small_unexplored(full_map, explored_map, area_thresh)

    # Blur the full_map with a kernel of 3
    blurred_map = cv2.blur(explored_map, (3, 3))
    # Threshold the blurred map to get a binary image
    _, blurred_map = cv2.threshold(blurred_map, 0, 255, cv2.THRESH_BINARY)
    # Make a three channel image by stacking all images
    stacks = np.stack((full_map, explored_map, blurred_map), axis=2)
    # Make a single-channel mask where pixels are white if they are:
    # 1). navigable (white in full_map);
    # 2). unexplored (black in explored_map), but also;
    # 3). adjacent to explored pixels (white in blurred_map)
    mask = np.all(stacks == [255, 0, 255], axis=2).astype(np.uint8) * 255

    return mask


def filter_out_small_unexplored(
    full_map: np.ndarray, explored_map: np.ndarray, area_thresh: int
):
    """Edit the explored map to add small unexplored areas"""
    # Make mask of unexplored areas
    stacks = np.stack((full_map, explored_map), axis=2)
    unexplored_mask = np.all(stacks == [255, 0], axis=2).astype(np.uint8) * 255
    # Find contours in the unexplored mask
    contours, _ = cv2.findContours(
        unexplored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Add small unexplored areas to the explored map
    for contour in contours:
        if cv2.contourArea(contour) < area_thresh:
            cv2.drawContours(explored_map, [contour], -1, 255, -1)


def frontier_waypoints(frontier_mask: np.ndarray) -> List[np.ndarray]:
    """Finds waypoints in a frontier mask.

    Args:
        frontier_mask (np.ndarray): A mono-channel mask where white contours represent
        each frontier.

    Returns:
        np.ndarray: A list of waypoints in the frontier mask.
    """
    # Find contours in the frontier mask
    contours, _ = cv2.findContours(
        frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Find the center of each contour
    waypoints = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            waypoints.append(np.array([x, y]))
    return waypoints


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
