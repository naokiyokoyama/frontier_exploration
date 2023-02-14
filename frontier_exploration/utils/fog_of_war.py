import cv2
import numpy as np
from frontier_exploration.utils.general_utils import wrap_heading


def get_two_farthest_points(source, cnt):
    """Returns the two points in the contour cnt that form the smallest and largest
    angles from the source point."""
    pts = cnt.reshape(-1, 2)
    angles = np.arctan2(pts[:, 1] - source[1], pts[:, 0] - source[0])
    # Get the two points that form the smallest and largest angles from the source
    min_idx = np.argmin(angles)
    max_idx = np.argmax(angles)
    return cnt[min_idx], cnt[max_idx]


def vectorize_get_line_points(current_point, points, max_line_len):
    angles = np.arctan2(
        points[..., 1] - current_point[1], points[..., 0] - current_point[0]
    )
    endpoints = np.stack(
        (
            points[..., 0] + max_line_len * np.cos(angles),
            points[..., 1] + max_line_len * np.sin(angles),
        ),
        axis=-1,
    )
    endpoints = endpoints.astype(np.int32)

    line_points = np.stack([points.reshape(-1, 2), endpoints.reshape(-1, 2)], axis=1)
    return line_points


def get_line_points(current_point, points, maxlen):
    current_point = np.repeat(current_point[np.newaxis, :], 2 * len(points), axis=0)
    points = np.repeat(points, 2, axis=0)
    diffs = current_point - points
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    end_points = current_point + maxlen * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    line_points = np.concatenate((points, end_points), axis=1)
    line_points = np.array(line_points, dtype=np.int32)
    return line_points


def reveal_fog_of_war(
    top_down_map: np.ndarray,
    current_fog_of_war_mask: np.ndarray,
    current_point: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
) -> np.ndarray:
    curr_pt_cv2 = current_point[::-1].astype(int)
    angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi/2))

    cone_mask = cv2.ellipse(
        np.zeros_like(top_down_map),
        curr_pt_cv2,
        (int(max_line_len), int(max_line_len)),
        0,
        angle_cv2 - fov / 2,
        angle_cv2 + fov / 2,
        1,
        -1,
    )

    # Create a mask of pixels that are both in the cone and NOT in the top_down_map
    obstacles_in_cone = cv2.bitwise_and(cone_mask, 1 - top_down_map)

    # Find the contours of the obstacles in the cone
    obstacle_contours, _ = cv2.findContours(
        obstacles_in_cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(obstacle_contours) == 0:
        return current_fog_of_war_mask  # there were no obstacles in the cone

    # Find the two points in each contour that form the smallest and largest angles
    # from the current position
    points = np.array(
        [get_two_farthest_points(curr_pt_cv2, cnt) for cnt in obstacle_contours]
    ).reshape((-1, 2, 2))

    # Fragment the cone using obstacles and two lines per obstacle in the cone
    visible_cone_mask = cv2.bitwise_and(cone_mask, top_down_map)
    line_points = vectorize_get_line_points(curr_pt_cv2, points, max_line_len * 1.05)
    # Draw all lines simultaneously using cv2.polylines
    cv2.polylines(visible_cone_mask, line_points, isClosed=False, color=0, thickness=2)

    # Identify the contour that is closest to the current position
    final_contours, _ = cv2.findContours(
        visible_cone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    visible_area = None
    min_dist = np.inf
    for cnt in final_contours:
        pt = tuple([int(i) for i in curr_pt_cv2])
        dist = abs(cv2.pointPolygonTest(cnt, pt, True))
        if dist < min_dist:
            min_dist = dist
            visible_area = cnt
    if min_dist > 3:
        return current_fog_of_war_mask  # the closest contour was too far away

    new_fog = cv2.drawContours(current_fog_of_war_mask, [visible_area], 0, 1, -1)
    return new_fog


if __name__ == "__main__":
    import time

    SHOW = True  # whether to imshow the results
    window_size = 1000
    N = 100
    L = (20, 50)
    max_line_len = 500
    fov = 90
    agent_radius = 20
    blank = np.ones((window_size, window_size), dtype=np.uint8)
    times = []
    for _ in range(500):
        t_start = time.time()
        top_down_map = blank.copy()
        # Populate the image with N random rectangles, with a (min, max) length of L
        for _ in range(N):
            rect_0 = np.random.randint(0, window_size, 2)
            rect_1 = rect_0 + np.random.randint(*L, 2)
            cv2.rectangle(top_down_map, rect_0, rect_1, 0, -1)
        # Sample random position and heading
        current_point = np.random.randint(window_size * 0.25, window_size * 0.75, 2)
        # Re-sample current_point if it is inside an obstacle
        while top_down_map[current_point[1], current_point[0]] != 1:
            current_point = np.random.randint(window_size * 0.25, window_size * 0.75, 2)
        current_angle = np.random.uniform(-np.pi, np.pi)

        fog = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=np.zeros_like(top_down_map),
            current_point=current_point,
            current_angle=current_angle,
            fov=fov,
            max_line_len=max_line_len,
        )

        times.append(time.time() - t_start)

        if SHOW:
            viz = np.ones((window_size, window_size, 3), dtype=np.uint8) * np.array(
                (60, 60, 60), dtype=np.uint8
            )
            viz[top_down_map == 0] = (255, 255, 255)
            viz[fog > 0] = (127, 127, 127)
            cv2.circle(viz, current_point[::-1], agent_radius, (255, 192, 15), -1)

            heading_end_pt = (
                agent_radius
                * 1.4
                * np.array([np.sin(current_angle), np.cos(current_angle)])
            ) + current_point[::-1]

            # Draw a line from the current position showing the current_angle
            cv2.line(
                viz,
                current_point[::-1],
                (int(heading_end_pt[0]), int(heading_end_pt[1])),
                (0, 0, 0),
                max(1, agent_radius // 4),
            )
            cv2.imshow("viz", viz)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    print(f"Average time: {np.mean(times[1:])}")
