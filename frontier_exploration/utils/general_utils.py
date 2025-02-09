import cv2
import numpy as np
from numba import njit


@njit
def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi


def handle_single_point(func):
    def wrapper(coords):
        # Check if coords is a single point (1D array) or multiple points (2D array)
        if len(coords.shape) == 1:
            # If it's a single point, add an extra dimension to make it a 2D array
            coords = np.expand_dims(coords, axis=0)
            # Call the original function and remove the extra dimension from the result
            return np.squeeze(func(coords))
        else:
            # If multiple points, just call the original function
            return func(coords)

    return wrapper


@handle_single_point
def habitat_to_xyz(coords: np.ndarray) -> np.ndarray:
    return np.array([-coords[:, 2], -coords[:, 0], coords[:, 1]]).T


@handle_single_point
def xyz_to_habitat(coords: np.ndarray) -> np.ndarray:
    return np.array([-coords[:, 1], coords[:, 2], -coords[:, 0]]).T


def images_to_video(image_list, output_path, fps=5):
    # Get the height and width from the first image
    height, width = image_list[0].shape[:2]
    print(f"Creating video with dimensions {width}x{height} at {fps} FPS")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image in image_list:
        out.write(image)

    # Release the VideoWriter
    out.release()

    print(f"Video saved to {output_path}")


def interpolate_path(points: np.ndarray, max_dist: float) -> np.ndarray:
    """
    Interpolate points along a path such that no two consecutive points are further
    apart than max_dist.

    Args:
        points: numpy array of shape (N, 2) or (N, 3) containing coordinates
        max_dist: maximum allowed Euclidean distance between consecutive points

    Returns:
        numpy array containing original and interpolated points, maintaining max_dist
        constraint

    Raises:
        ValueError: if points array is not of shape (N, 2) or (N, 3)
    """
    if not (points.ndim == 2 and points.shape[1] in (2, 3)):
        raise ValueError("Points array must be of shape (N, 2) or (N, 3)")

    # Initialize list to store all points (original + interpolated)
    result = [points[0]]

    # Process each pair of consecutive points
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        # Calculate distance between points
        dist = np.linalg.norm(p2 - p1)

        if dist > max_dist:
            # Calculate number of segments needed
            num_segments = int(np.ceil(dist / max_dist))

            # Create interpolated points
            for j in range(1, num_segments):
                t = j / num_segments
                interp_point = p1 + t * (p2 - p1)
                result.append(interp_point)

        # Add the second point of the pair
        result.append(p2)

    return np.array(result)
