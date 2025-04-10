import glob
import json
import os
import os.path as osp
import pickle
import queue
import random
import signal
import sys
import threading
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import tqdm

from frontier_exploration.utils.general_utils import sliding_window_strided
from frontier_exploration.utils.sim_utils import load_reshaped_npy

PURGE_COUNT = 0
# Add a set to track already processed or in-progress episodes
PROCESSED_EPISODES = set()
# Add a lock for thread-safe operations on shared resources
LOCK = threading.Lock()


# Add a global flag to indicate termination request
TERMINATE = False


def signal_handler(sig, frame):
    """
    Handle termination signals by setting the TERMINATE flag and initiating cleanup
    """
    global TERMINATE
    signal_name = signal.Signals(sig).name
    print(f"\nReceived {signal_name} signal. Initiating shutdown...")
    TERMINATE = True
    # Don't exit immediately - let the main loop handle the termination process


def cleanup():
    """
    Clean up resources and ensure all tasks complete before exiting
    """
    print("Cleaning up resources...")

    # Wait for the queue to be empty with a timeout
    remaining_tasks = task_queue.unfinished_tasks
    if remaining_tasks:
        print(f"Waiting for {remaining_tasks} tasks to complete...")
        try:
            # Set a timeout for joining the queue to avoid hanging indefinitely
            # This approach requires a custom implementation since queue.join() doesn't accept a timeout
            wait_start = time.time()
            wait_timeout = 60  # 60 seconds timeout
            while (
                task_queue.unfinished_tasks and time.time() - wait_start < wait_timeout
            ):
                time.sleep(0.5)

            if task_queue.unfinished_tasks:
                print(
                    f"Timeout reached with {task_queue.unfinished_tasks} tasks"
                    " remaining"
                )
        except Exception as e:
            print(f"Error during queue cleanup: {e}")

    print("All possible tasks completed. Shutting down workers...")

    if args.purge:
        print(f"Purged {PURGE_COUNT} files")

    # Exit with appropriate code
    sys.exit(0)


def check_episode_dir_completed(episode_dir: str) -> bool:
    min_length = int(os.environ["MIN_LENGTH"])
    max_length = int(os.environ["MAX_LENGTH"])
    output = osp.join(
        episode_dir,
        f"valid_start_end_{min_length}_{max_length}{os.environ['FRONTIER_TYPE']}.pkl",
    )
    return should_stop(output)


def process_sample(episode_dir: str, purge: bool) -> None:
    exploration_imgs_dir = osp.join(episode_dir, "exploration_imgs_0")

    min_length = int(os.environ["MIN_LENGTH"])
    max_length = int(os.environ["MAX_LENGTH"])
    output = osp.join(
        episode_dir,
        f"valid_start_end_{min_length}_{max_length}{os.environ['FRONTIER_TYPE']}.pkl",
    )

    if purge:
        global PURGE_COUNT
        if osp.exists(output):
            os.remove(output)
            PURGE_COUNT += 1
        return

    if should_stop(output):
        return

    npys = []
    for path in [exploration_imgs_dir, episode_dir]:
        pattern = osp.join(path, "*.npy")
        matches = glob.glob(pattern)
        assert (
            len(matches) == 1
        ), f"Expected 1 npy file, got {len(matches)} for pattern {pattern}"
        npys.append(matches[0])
    exploration_npy, frontiers_npy = npys
    exploration_masks = load_reshaped_npy(exploration_npy)
    frontier_masks = load_reshaped_npy(frontiers_npy)

    json_path = next(glob.iglob(episode_dir + "/*.json"))
    with open(json_path) as f:
        sample = json.load(f)
    # Maps from string ("gt|a,b,c,d") to list of integers (timesteps)
    f_dict: Dict[str, List[int]] = sample["all_frontiers" + os.environ["FRONTIER_TYPE"]]
    # Maps from timestep to string ("gt|a,b,c,d")
    frontier_sets: List[str] = []
    for k, v in f_dict.items():
        candidate_timesteps = [i for i in v if i > 8]
        if not candidate_timesteps:
            continue
        frontier_sets.append(k)

    if should_stop(output):
        return

    # Touch the file to indicate that we are processing it
    with open(output, "w"):
        pass

    if not frontier_sets:
        return

    try:
        cache_valid_frontier_set(
            output,
            frontier_sets,
            exploration_masks,
            frontier_masks,
            min_length,
            max_length,
        )
    except Exception as e:
        # Delete the file if it is empty
        if osp.exists(output) and osp.getsize(output) == 0:
            os.remove(output)
        raise e


def should_stop(output: str) -> bool:
    # If the file already exists:
    # - Check if its empty:
    #   - If so, delete it only if it wasn't modified recently (2 min)
    #   - Otherwise, return
    # - If it is not empty, return
    if osp.exists(output):
        try:
            size = osp.getsize(output)
            mtime = osp.getmtime(output)
        except FileNotFoundError:
            return False
        if size == 0:
            if time.time() - mtime > 120:
                try:
                    os.remove(output)
                except FileNotFoundError:
                    pass
            else:
                return True
        else:
            return True
    return False


def cache_valid_frontier_set(
    output: str,
    frontier_sets: List[str],
    exploration_masks: np.ndarray,
    frontier_masks: np.ndarray,
    min_length: int,
    max_length: int,
    coverage_thresh: float = 0.2,
) -> None:
    cache: Dict[str, Dict[int, Tuple[Union[Tuple[int, int], int], ...]]] = {}
    min_tour = min(min_length, exploration_masks.shape[0])
    max_tour = min(max_length, exploration_masks.shape[0])
    for frontier_set in frontier_sets:
        cache[frontier_set] = {}
        curr_fids = list(map(int, frontier_set.split("|")[1].split(",")))
        sub_frontier_masks = frontier_masks[curr_fids]
        overlaps = compute_mask_overlap(
            sub_frontier_masks, exploration_masks, threshold=coverage_thresh
        )
        for window_size in range(min_tour, max_tour + 1):
            windows = sliding_window_strided(overlaps, window_size)
            valid_windows = windows.any(axis=1).sum(axis=1) >= len(curr_fids) - 1
            if not valid_windows.any():
                continue
            valid_window_indices = np.where(valid_windows)[0]
            cache[frontier_set][window_size] = compress_int_list(
                valid_window_indices.tolist()
            )
    if len(cache) > 0:
        with open(output, "wb") as f:
            pickle.dump(cache, f)
            print(f"Saved {output}")


def calculate_overlap(reference: np.ndarray, channels: np.ndarray) -> np.ndarray:
    """
    Calculate the percentage of overlapping pixels between a reference binary array
    and each channel of a multi-channel binary array.

    Args:
        reference: Binary numpy array of shape (H, W)
        channels: Binary numpy array of shape (N, H, W)

    Returns:
        A numpy array of shape (N,) where the i-th element represents the percentage
        of overlapping pixels between the reference array and the i-th channel,
        calculated as (number of overlapping set pixels) / (total set pixels in i-th channel)

    Notes:
        - Both input arrays should contain only binary values (0 and 1)
        - If a channel has no set pixels (all zeros), the function returns 0 for that channel
    """
    # Check input shapes compatibility
    if reference.shape != channels.shape[1:]:
        raise ValueError(
            f"Shape mismatch: reference shape {reference.shape} is not compatible "
            f"with channels shape {channels.shape}"
        )

    # Check if arrays are binary
    if not np.all(np.isin(reference, [0, 1])) or not np.all(np.isin(channels, [0, 1])):
        raise ValueError("Input arrays must be binary (contain only 0 and 1 values)")

    # Calculate intersection (logical AND) between reference and each channel
    intersection = reference[np.newaxis, :, :] & channels

    # Count the number of set pixels in each channel
    channel_set_pixels = np.sum(channels, axis=(1, 2))

    # Count the number of overlapping pixels in each channel
    overlap_pixels = np.sum(intersection, axis=(1, 2))

    # Calculate the overlap percentage, avoiding division by zero
    overlap_percentage = np.zeros(channels.shape[0], dtype=float)
    non_zero_mask = channel_set_pixels > 0
    overlap_percentage[non_zero_mask] = (
        overlap_pixels[non_zero_mask] / channel_set_pixels[non_zero_mask]
    )

    return overlap_percentage


def worker_function(task_queue, purge, worker_id):
    """
    Worker function that processes episodes from the queue
    """
    print(f"Worker {worker_id} started")
    while not TERMINATE:  # Check the termination flag
        try:
            # Use a short timeout so workers can check the termination flag frequently
            episode_dir = task_queue.get(timeout=1)
            try:
                process_sample(episode_dir, purge=purge)
            except Exception as e:
                import traceback

                print(
                    f"Worker {worker_id} encountered error processing"
                    f" {episode_dir}: {e}"
                )
                print(traceback.format_exc())
            finally:
                task_queue.task_done()
                # Mark this episode as processed
                with LOCK:
                    PROCESSED_EPISODES.add(episode_dir)
        except queue.Empty:
            # No more tasks in queue for now, just continue the loop
            # This will now check the termination flag every second
            continue
        except Exception as e:
            import traceback

            print(f"Worker {worker_id} encountered unexpected error: {e}")
            print(traceback.format_exc())

    print(f"Worker {worker_id} shutting down")


def find_new_episodes(input_dirs):
    """
    Find all episode directories that have not been processed yet
    """
    new_episodes = []
    for input_dir in tqdm.tqdm(input_dirs):
        episodes = glob.glob(input_dir + "/**/episode_*/", recursive=True)
        for episode_dir in tqdm.tqdm(episodes):
            # Skip already completed or in-progress episodes
            if episode_dir in PROCESSED_EPISODES or check_episode_dir_completed(
                episode_dir
            ):
                continue
            new_episodes.append(episode_dir)
    return new_episodes


def compute_mask_overlap(
    masks1: np.ndarray, masks2: np.ndarray, threshold: float = 0.2
) -> np.ndarray:
    """
    Compute overlap between two sets of binary masks.

    Args:
        masks1 (np.ndarray): Binary array of shape (N, H, W)
        masks2 (np.ndarray): Binary array of shape (M, H, W)
        threshold (float, optional): Minimum percentage of overlap required, default 0.2
                                    (20%)

    Returns:
        np.ndarray: Binary array of shape (M, N) where element (m, n) is True if:
                    1. The n-th mask in masks1 has at least one set pixel
                    2. The percentage of set pixels in n-th mask of masks1 that overlap
                       with set pixels in m-th mask of masks2 is above threshold
    """
    # Ensure inputs are binary
    masks1 = masks1.astype(bool)
    masks2 = masks2.astype(bool)

    # Get the dimensions
    N, H, W = masks1.shape
    M = masks2.shape[0]

    # Count set pixels in each mask from masks1
    mask1_areas = masks1.reshape(N, -1).sum(axis=1)  # Shape: (N,)

    # Create a matrix to store results
    overlap_matrix = np.zeros((M, N), dtype=bool)

    # Only proceed with masks that have non-zero areas
    valid_mask_indices = np.where(mask1_areas > 0)[0]

    if len(valid_mask_indices) > 0:
        # Reshape masks1 for vectorized computation
        masks1_valid = masks1[valid_mask_indices].reshape(
            len(valid_mask_indices), 1, H, W
        )

        # Compute intersection in a vectorized way
        # Broadcast masks1_valid to (valid_N, M, H, W) and masks2 to (1, M, H, W)
        intersection = masks1_valid & masks2.reshape(1, M, H, W)

        # Sum over the spatial dimensions to get intersection area
        intersection_areas = intersection.reshape(len(valid_mask_indices), M, -1).sum(
            axis=2
        )  # Shape: (valid_N, M)

        # Calculate percentage of overlap
        # Reshape mask1_areas for broadcasting
        mask1_areas_valid = mask1_areas[valid_mask_indices].reshape(
            -1, 1
        )  # Shape: (valid_N, 1)

        # Compute percentage of masks1 pixels that overlap with masks2
        overlap_percentage = (
            intersection_areas / mask1_areas_valid
        )  # Shape: (valid_N, M)

        # Apply threshold
        valid_overlaps = overlap_percentage > threshold  # Shape: (valid_N, M)

        # Store results in the appropriate indices
        # Need to transpose since we need output of shape (M, N)
        overlap_matrix[:, valid_mask_indices] = valid_overlaps.T

    return overlap_matrix


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


def monitor_progress(input_dirs, check_interval):
    """
    Monitor and display progress of episode processing using a progress bar
    """
    pbar = None
    processed_count = 0

    while True:
        # Get all episode directories
        all_episodes = []
        for input_dir in input_dirs:
            episodes = glob.glob(input_dir + "/**/episode_*/", recursive=True)
            all_episodes.extend(episodes)

        total_episodes = len(all_episodes)

        # Count completed episodes
        completed_count = sum(
            1 for ep in all_episodes if check_episode_dir_completed(ep)
        )

        # Initialize or update progress bar
        if pbar is None:
            pbar = tqdm.tqdm(total=total_episodes, desc="Processing episodes")
            pbar.update(completed_count)
            processed_count = completed_count
        elif completed_count > processed_count:
            pbar.update(completed_count - processed_count)
            processed_count = completed_count
        elif total_episodes > pbar.total:
            # New episodes detected, update total
            pbar.total = total_episodes
            pbar.refresh()

        time.sleep(check_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+")
    parser.add_argument("--frontier-type", type=str, default="")
    parser.add_argument("--min-len", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--purge", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes to use for parallel processing",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=15,
        help="Interval in seconds to check for new episodes",
    )
    parser.add_argument(
        "--run-time",
        type=int,
        default=0,
        help="Total run time in seconds (0 for infinite)",
    )
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Only monitor progress without processing",
    )
    args = parser.parse_args()

    os.environ["FRONTIER_TYPE"] = args.frontier_type
    os.environ["MIN_LENGTH"] = str(args.min_len)
    os.environ["MAX_LENGTH"] = str(args.max_len)
    os.environ["DONT_VALIDATE"] = "0"

    if args.monitor_only:
        print("Starting progress monitoring...")
        try:
            monitor_progress(args.input_dirs, args.check_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        sys.exit(0)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Keyboard interrupt (Ctrl+C)
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal (kill, SLURM)
    print("Starting dynamic episode processing...")

    # Create a task queue for episode directories
    task_queue = queue.Queue()

    # Create worker threads
    num_workers = max(1, args.num_workers)
    workers = []

    for i in range(num_workers):
        worker = threading.Thread(
            target=worker_function, args=(task_queue, args.purge, i), daemon=True
        )
        workers.append(worker)
        worker.start()

    # Main processing loop
    start_time = time.time()
    try:
        while not TERMINATE:  # Check termination flag
            # Check if we've reached the specified run time
            if args.run_time > 0 and time.time() - start_time > args.run_time:
                print(
                    f"Reached specified run time of {args.run_time} seconds. Exiting."
                )
                break

            # Find new episodes
            print("Scanning for new episode directories...")
            new_episodes = find_new_episodes(args.input_dirs)

            if new_episodes:
                random.shuffle(new_episodes)
                print(f"Found {len(new_episodes)} new episode directories")

                # Add new episodes to the queue
                for episode_dir in new_episodes:
                    if TERMINATE:  # Check termination flag before adding new tasks
                        break
                    task_queue.put(episode_dir)

                # Wait for current batch to complete
                if not TERMINATE:  # Only wait if not terminating
                    task_queue.join()
            else:
                print(
                    f"No new episodes found. Waiting {args.check_interval} seconds..."
                )
                # Sleep in smaller increments to check termination flag more frequently
                for _ in range(args.check_interval):
                    if TERMINATE:
                        break
                    time.sleep(1)

    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
        import traceback

        print(traceback.format_exc())

    finally:
        if not TERMINATE:
            # If we're here but TERMINATE is not set, set it now
            TERMINATE = True

        # Call the cleanup function
        cleanup()
