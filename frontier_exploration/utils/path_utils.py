import numpy as np
from numba import njit

from frontier_exploration.utils.general_utils import wrap_heading


@njit
def completion_time_heuristic(
    sim_waypoints, agent_position, agent_heading, lin_vel, ang_vel
):
    """ An admissible, consistent heuristic for finding the completion time of an
    obstacle-free path to a waypoint. For each waypoint, this returns the shortest
    amount of time it would take to reach that waypoint from the agent's current
    position and heading assuming NO obstacles, serving as a lower bound on the actual
    completion time. Assumes agent has point-turn dynamics."""
    euclidean_dists = euclidean_heuristic(sim_waypoints, agent_position)
    heading_to_waypoints = np.arctan2(
        sim_waypoints[:, 2] - agent_position[2],
        sim_waypoints[:, 0] - agent_position[0],
    )
    heading = wrap_heading(np.pi / 2.0 - agent_heading)
    heading_errors = np.abs(wrap_heading(heading_to_waypoints - heading))
    completion_times = heading_errors / ang_vel + euclidean_dists / lin_vel
    return completion_times


@njit
def euclidean_heuristic(sim_waypoints, agent_position):
    """ An admissible, consistent heuristic for finding the length of the shortest
    obstacle-free path to a waypoint. For each waypoint, this returns its euclidean
    distance from the agent's position, serving as a lower bound on the actual
    length of the shortest obstacle-free path."""
    euclid_dists = np.sqrt(
        (sim_waypoints[:, 2] - agent_position[2]) ** 2
        + (sim_waypoints[:, 0] - agent_position[0]) ** 2
    )
    return euclid_dists


@njit
def shortest_path_completion_time(path, max_lin_vel, max_ang_vel, yaw_diff):
    time = 0
    cur_pos = path[0]
    cur_yaw = None
    for i in range(1, path.shape[0]):
        target_pos = path[i]
        target_yaw = np.arctan2(target_pos[2] - cur_pos[2], target_pos[0] - cur_pos[0])

        distance = np.sqrt(
            (target_pos[2] - cur_pos[2]) ** 2 + (target_pos[0] - cur_pos[0]) ** 2
        )
        if cur_yaw is not None:
            yaw_diff = np.abs(wrap_heading(target_yaw - cur_yaw))

        lin_time = distance / max_lin_vel
        ang_time = yaw_diff / max_ang_vel
        time += lin_time + ang_time

        cur_pos = target_pos
        cur_yaw = target_yaw

    return time
