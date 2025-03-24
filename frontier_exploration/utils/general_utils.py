from typing import List, Tuple, Union

import cv2
import numpy as np
import quaternion as qt
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)
from numba import njit
from numpy.lib.stride_tricks import as_strided


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


def calculate_perpendicularity(p1: np.ndarray, line_pairs: np.ndarray) -> np.ndarray:
    """
    Calculate how perpendicular multiple line pairs are to a point.

    Parameters:
    p1: np.array([x, y]) - The standalone point
    line_pairs: np.array([[x1, y1], [x2, y2]]) shape (N,2,2) - Array of line pairs
                where each pair consists of two points

    Returns:
    np.array shape (N,) - Array of perpendicularity scores
        0 means perfectly perpendicular, 1 means parallel
    """
    # Calculate midpoints for all line pairs at once
    midpoints = line_pairs.mean(axis=1)  # Shape: (N,2)

    # Calculate direction vectors for all line pairs
    v1 = line_pairs[:, 1] - line_pairs[:, 0]  # Shape: (N,2)
    v2 = midpoints - p1  # Shape: (N,2)

    # Calculate dot products for all pairs
    dot_products = np.sum(v1 * v2, axis=1)  # Shape: (N,)

    # Calculate norms
    v1_norms = np.linalg.norm(v1, axis=1)  # Shape: (N,)
    v2_norms = np.linalg.norm(v2, axis=1)  # Shape: (N,)

    # Return normalized absolute dot products
    return np.abs(dot_products / (v1_norms * v2_norms))


def polar_angle_to_quaternion(phi: float) -> qt.quaternion:
    """
    Converts a polar angle (phi) back to a quaternion.

    This function is the inverse of get_polar_angle(). It creates a quaternion
    that represents a rotation around the y-axis such that when processed by
    get_polar_angle(), it will return the original phi.

    Args:
        phi: The polar angle in radians

    Returns:
        quaternion.quaternion: A quaternion representing the rotation
    """
    # Based on our analysis, there's a π difference between the input angle
    # and the angle returned by get_polar_angle. We need to adjust for this.
    adjusted_angle = phi + np.pi

    # Now create the quaternion for a rotation around the y-axis
    half_angle = adjusted_angle / 2

    # Create quaternion in format [w, x, y, z]
    w = np.cos(half_angle)
    x = 0
    y = np.sin(half_angle)
    z = 0

    # Return in quaternion format required by the library
    return qt.quaternion(w, x, y, z)


def get_polar_angle(rotation: np.ndarray) -> float:
    # quaternion is in x, y, z, w format
    ref_rotation = rotation
    heading_vector = quaternion_rotate_vector(
        ref_rotation.inverse(), np.array([0, 0, -1])  # noqa
    )
    phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]
    return float(phi)


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the angle formed by three 2D points where p2 is the vertex.

    This function computes the angle in radians between vectors p1-p2 and p3-p2,
    where p2 is the vertex of the angle. The calculation uses the dot product
    formula to determine the angle.

    Args:
        p1: The first point as a numpy array of shape (2,)
        p2: The vertex point as a numpy array of shape (2,)
        p3: The third point as a numpy array of shape (2,)

    Returns:
        float: The angle in radians between the vectors
    """
    # Calculate vectors from the vertex to the other points
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate the angle using the dot product formula
    # cos(θ) = (v1 · v2) / (|v1| * |v2|)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Handle potential division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    # Calculate cosine of the angle, ensuring it's within valid range [-1, 1]
    cos_angle = np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)

    # Return the angle in radians
    return np.arccos(cos_angle)


def sliding_window_strided(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Create a sliding window view of a 2D array using stride_tricks.

    Args:
        arr (np.ndarray): Input array of shape (N, M)
        k (int): Window size, must be less than N

    Returns:
        np.ndarray: Array of shape (X, K, M) where X = N - K + 1
    """
    n, m = arr.shape
    s0, s1 = arr.strides
    x = n - k + 1

    return as_strided(arr, shape=(x, k, m), strides=(s0, s0, s1), writeable=False)


def compress_int_list(nums: List[int]) -> Tuple[Union[Tuple[int, int], int], ...]:
    """
    Compresses a sorted list of unique integers into a tuple of ranges and individual
    values.

    Args:
        nums: A list of unique integers

    Returns:
        A tuple where each element is either:
        - A tuple (i, j) representing a consecutive range from i to j (inclusive)
        - An integer representing a single value
    """
    assert len(nums) == len(set(nums))
    if not nums:
        return tuple()

    nums = sorted(nums)

    result = []
    start = nums[0]
    end = nums[0]

    for i in range(1, len(nums)):
        if nums[i] == end + 1:
            # Extend the current range
            end = nums[i]
        else:
            # Add the previous range or individual element
            if start == end:
                result.append(start)
            else:
                result.append((start, end))

            # Start a new range
            start = nums[i]
            end = nums[i]

    # Add the last range or individual element
    if start == end:
        result.append(start)
    else:
        result.append((start, end))

    return tuple(result)
