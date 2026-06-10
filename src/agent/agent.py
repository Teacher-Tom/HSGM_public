import json
import math
import re
from abc import ABC, abstractmethod

import PIL.Image

from simWrapper import SimWrapper, PolarAction
from mapper import Instruct_Mapper
import time
from typing import Optional, List
from PIL.Image import Image
from utils import *
from scipy.spatial.transform import Rotation as R

import socket

def _setup_proxy(preferred_url: str = "http://127.0.0.1:7897") -> None:
    """Set proxy only when:
    1. User has NOT already set http_proxy/https_proxy env vars
    2. The preferred proxy is TCP-reachable (otherwise skip)
    """
    if os.environ.get("http_proxy") or os.environ.get("https_proxy"):
        return
    try:
        host, port = preferred_url.replace("http://", "").replace("/", "").split(":")
        with socket.create_connection((host, int(port)), timeout=1.0):
            os.environ["http_proxy"] = preferred_url
            os.environ["https_proxy"] = preferred_url
    except Exception:
        pass

_setup_proxy()

def draw_direction_markers(img: np.ndarray) -> np.ndarray:
    """
    Draw red direction triangles (top/bottom/left/right) with F/B/L/R labels (auto-scaled).
    """
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return img

    s = max(6, int(min(h, w) * 0.025))  # triangle half-size (shrunk)
    m = max(2, int(s * 0.3))  # margin
    border_th = max(1, s // 8)  # stroke thickness
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_triangle_with_label(bg, center, direction, label):
        cx, cy = center
        if direction == 'up':
            pts = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]], dtype=np.int32)
        elif direction == 'down':
            pts = np.array([[cx, cy + s], [cx - s, cy - s], [cx + s, cy - s]], dtype=np.int32)
        elif direction == 'left':
            pts = np.array([[cx - s, cy], [cx + s, cy - s], [cx + s, cy + s]], dtype=np.int32)
        else:  # right
            pts = np.array([[cx + s, cy], [cx - s, cy - s], [cx - s, cy + s]], dtype=np.int32)

        cv2.fillPoly(bg, [pts], (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.polylines(bg, [pts], isClosed=True, color=(0, 0, 0), thickness=border_th, lineType=cv2.LINE_AA)

        centroid = pts.mean(axis=0).astype(int)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        allow_w = max(8, int((x_max - x_min) * 0.55))
        allow_h = max(8, int((y_max - y_min) * 0.45))

        (tw1, th1), _ = cv2.getTextSize(label, font, 0.7, 3)
        if tw1 == 0 or th1 == 0:
            return
        scale = min(allow_w / tw1, allow_h / th1)
        scale = max(0.4, min(2.0, scale))  # clamp range
        thickness = max(1, int(scale * 1.2))

        (tw, th), bl = cv2.getTextSize(label, font, scale, thickness)

        text_org = (int(centroid[0] - tw / 2), int(centroid[1] + th / 2) - 1)

        cv2.putText(bg, label, text_org, font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)

    draw_triangle_with_label(img, (w // 2, m + s), 'up', 'F')

    draw_triangle_with_label(img, (w // 2, h - m - s), 'down', 'B')

    draw_triangle_with_label(img, (m + s, h // 2), 'left', 'L')

    draw_triangle_with_label(img, (w - m - s, h // 2), 'right', 'R')

    return img

def create_top_down_map_centered(
    mapper: Instruct_Mapper,
    fov_angle_deg: float,
    target_point=None,
    map_scale=0.025,
    map_size_px=1024,
    fov_range=1.0,
    show_all_labels=False,
    label_volume_threshold=100,
    crop_size_m=16.0,
    action_candidates=None,
    subtasks=None,
):
    """
    action_candidates: Dict[str, Dict], each entry example:
      "waypoint 1": {
        "type": "waypoint",
        "target_point_world": [x, y, z]
      }
    subtasks: Dict[str, Sequence[float]] subtask name -> world coordinate ([x,y,z]).
    """
    t_start = time.time()
    timings = {}

    t = time.time()
    agent_pos = mapper.current_position
    agent_rot = mapper.current_rotation

    forward_vec_3d = agent_rot @ np.array([0, 0, -1])
    forward_vec_2d = forward_vec_3d[:2]
    if np.linalg.norm(forward_vec_2d) < 1e-6:
        forward_vec_2d = np.array([0, -1])
    forward_vec_2d /= np.linalg.norm(forward_vec_2d)

    agent_angle_rad = np.arctan2(forward_vec_2d[1], forward_vec_2d[0])
    rotation_angle = -agent_angle_rad - np.pi / 2

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[c, -s], [s, c]])

    def transform_and_to_pixel(world_coords):
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)

        translated_coords = world_coords[:, :2] - agent_pos[:2]

        rotated_coords = translated_coords @ rotation_matrix.T

        pixel_coords = (rotated_coords / map_scale) + np.array([map_size_px / 2, map_size_px / 2])
        return pixel_coords.astype(int)

    timings['calc_transform'] = time.time() - t

    top_down_map = np.full((map_size_px, map_size_px, 3), 255, dtype=np.uint8)
    map_h, map_w = top_down_map.shape[:2]

    pcd_resolution = 0.025
    dilation_radius = max(1, int(pcd_resolution / map_scale / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    t = time.time()
    if not mapper.navigable_pcd.is_empty() and mapper.navigable_pcd.point.positions.shape[0] > 0:
        nav_pixels = transform_and_to_pixel(mapper.navigable_pcd.point.positions.cpu().numpy())
        valid_mask = (nav_pixels[:, 0] >= 0) & (nav_pixels[:, 0] < map_w) & (nav_pixels[:, 1] >= 0) & (
                    nav_pixels[:, 1] < map_h)
        if np.any(valid_mask):
            nav_mask = np.zeros((map_h, map_w), dtype=np.uint8)
            nav_mask[nav_pixels[valid_mask, 1], nav_pixels[valid_mask, 0]] = 255
            dilated_nav_mask = cv2.dilate(nav_mask, kernel)
            top_down_map[dilated_nav_mask == 255] = (200, 200, 200)
    timings['draw_navigable'] = time.time() - t

    t = time.time()
    obstacle_mask_bool = np.zeros((map_h, map_w), dtype=bool)
    if not mapper.obstacle_pcd.is_empty() and mapper.obstacle_pcd.point.positions.shape[0] > 0:
        obs_pixels = transform_and_to_pixel(mapper.obstacle_pcd.point.positions.cpu().numpy())
        valid_mask = (obs_pixels[:, 0] >= 0) & (obs_pixels[:, 0] < map_w) & (obs_pixels[:, 1] >= 0) & (
                    obs_pixels[:, 1] < map_h)
        if np.any(valid_mask):
            obs_mask = np.zeros((map_h, map_w), dtype=np.uint8)
            obs_mask[obs_pixels[valid_mask, 1], obs_pixels[valid_mask, 0]] = 255
            dilated_obs_mask = cv2.dilate(obs_mask, kernel)
            top_down_map[dilated_obs_mask == 255] = (50, 50, 50)
            obstacle_mask_bool = (dilated_obs_mask == 255)

            contours, _ = cv2.findContours(dilated_obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(top_down_map, contours, -1, (0, 0, 0), thickness=5)
    timings['draw_obstacle'] = time.time() - t

    t = time.time()
    if hasattr(mapper, 'frontier_pcd') and not mapper.frontier_pcd.is_empty():
        frontier_pixels = transform_and_to_pixel(mapper.frontier_pcd.point.positions.cpu().numpy())
        valid_mask = (frontier_pixels[:, 0] >= 0) & (frontier_pixels[:, 0] < map_w) & \
                     (frontier_pixels[:, 1] >= 0) & (frontier_pixels[:, 1] < map_h)
        valid_frontier_pixels = frontier_pixels[valid_mask]
        for pixel in valid_frontier_pixels:
            cv2.circle(top_down_map, tuple(pixel), 1, (0, 255, 255), -1)  # yellow dot
    timings['draw_frontier'] = time.time() - t

    t = time.time()
    if len(mapper.object_entities) > 0:
        for entity in mapper.object_entities:
            if entity['pcd'].point.positions.shape[0] > 0:
                obj_pixels = transform_and_to_pixel(entity['pcd'].point.positions.cpu().numpy())
                valid_pixels_mask = (obj_pixels[:, 0] >= 0) & (obj_pixels[:, 0] < map_w) & (obj_pixels[:, 1] >= 0) & (
                            obj_pixels[:, 1] < map_h)
                if not np.any(valid_pixels_mask): continue

                valid_obj_pixels = obj_pixels[valid_pixels_mask]
                intersection_mask = obstacle_mask_bool[valid_obj_pixels[:, 1], valid_obj_pixels[:, 0]]

                if np.any(intersection_mask):
                    final_pixels = valid_obj_pixels[intersection_mask]
                    if len(final_pixels) == 0: continue

                    obj_colors_rgb = entity['pcd'].point.colors.cpu().numpy()[valid_pixels_mask][intersection_mask]
                    object_color = (obj_colors_rgb[0] * 255)[::-1].astype(np.uint8)

                    object_mask_img = np.zeros((map_h, map_w), dtype=np.uint8)
                    object_mask_img[final_pixels[:, 1], final_pixels[:, 0]] = 255
                    fill_kernel = np.ones((5, 5), np.uint8)
                    filled_mask = cv2.morphologyEx(object_mask_img, cv2.MORPH_CLOSE, fill_kernel, iterations=2)
                    top_down_map[filled_mask == 255] = object_color
    timings['draw_objects'] = time.time() - t

    label_upscale_factor = 2
    if label_upscale_factor > 1:
        new_h, new_w = map_h * label_upscale_factor, map_w * label_upscale_factor
        top_down_map = cv2.resize(top_down_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        new_h, new_w = map_h, map_w

    t = time.time()
    drawn_labels = []
    label_merge_distance_px = 50
    for entity in sorted(mapper.object_entities, key=lambda e: e['class']):
        if not show_all_labels and entity['pcd'].point.positions.shape[0] < label_volume_threshold: continue
        if entity['pcd'].point.positions.shape[0] == 0: continue

        center_pixel = transform_and_to_pixel(entity['center'])[0]
        class_name = entity['class_name']

        should_merge = False
        for i, label_info in enumerate(drawn_labels):
            if label_info['class'] == class_name and np.linalg.norm(
                    center_pixel - label_info['center_pixel']) < label_merge_distance_px:
                drawn_labels[i]['center_pixel'] = (label_info['center_pixel'] + center_pixel) / 2
                should_merge = True
                break
        if not should_merge:
            drawn_labels.append({'class': class_name, 'center_pixel': center_pixel})

    for label_info in drawn_labels:
        center_pixel_scaled = (label_info['center_pixel'] * label_upscale_factor).astype(int)
        if not (0 <= center_pixel_scaled[0] < new_w and 0 <= center_pixel_scaled[1] < new_h): continue

        class_name = label_info['class']
        font_scale, font_thickness = 0.7, 1
        (tw, th), bl = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        box_tl = (center_pixel_scaled[0] - tw // 2 - 3, center_pixel_scaled[1] - th - 3)
        box_br = (center_pixel_scaled[0] + tw // 2 + 3, center_pixel_scaled[1] + bl)

        alpha = 0.8
        overlay = top_down_map.copy()
        cv2.rectangle(overlay, box_tl, box_br, (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, top_down_map, 1 - alpha, 0, top_down_map)
        cv2.rectangle(top_down_map, box_tl, box_br, (0, 0, 0), 1)

        text_org = (box_tl[0] + 3, box_br[1] - bl)
        cv2.putText(top_down_map, class_name, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), font_thickness, cv2.LINE_AA)
    timings['draw_labels'] = time.time() - t

    t = time.time()
    if len(mapper.trajectory_position) > 1:
        traj_pixels = transform_and_to_pixel(np.array(mapper.trajectory_position[-150:]))
        valid_traj_pixels = (traj_pixels * label_upscale_factor).astype(int)
        overlay = top_down_map.copy()
        cv2.polylines(overlay, [valid_traj_pixels], isClosed=False, color=(0, 0, 255),
                      thickness=2 * label_upscale_factor)
        alpha = 0.5  # trajectory alpha, adjustable
        top_down_map = cv2.addWeighted(overlay, alpha, top_down_map, 1 - alpha, 0)
    timings['draw_trajectory'] = time.time() - t

    # waypoints = mapper.get_candidate_waypoints(min_distance=0.2, max_distance=1.0, waypoint_grid_resolution=1.0)

    # if waypoints.shape[0] > 0:
    #     waypoint_pixels = transform_and_to_pixel(waypoints)
    #     waypoint_pixels_scaled = (waypoint_pixels * label_upscale_factor).astype(int)
    #     for pixel in waypoint_pixels_scaled:
    #         if 0 <= pixel[0] < new_w and 0 <= pixel[1] < new_h:

    #
    t = time.time()
    if target_point is not None:
        target_pixel = transform_and_to_pixel(np.array(target_point))[0]
        target_pixel_scaled = (target_pixel * label_upscale_factor).astype(int)

        if (0 <= target_pixel_scaled[0] < new_w) and (0 <= target_pixel_scaled[1] < new_h):

            cv2.circle(top_down_map, tuple(target_pixel_scaled), 8, (0, 255, 0), -1)  # green filled circle
            cv2.circle(top_down_map, tuple(target_pixel_scaled), 8, (0, 0, 0), 2)  # black border
    timings['draw_target'] = time.time() - t

    new_h, new_w = top_down_map.shape[:2]  # use resized dims if already resized

    if action_candidates:

        overlay = top_down_map.copy()

        for text, cand in action_candidates.items():
            if cand.get('type') != 'waypoint':
                continue  # only handle waypoint type for now

            try:
                idx = int(text.split()[-1])
            except (ValueError, IndexError):
                continue  # skip if text format mismatches

            coord = cand.get('target_point_world')
            if coord is None:
                continue

            pix = transform_and_to_pixel(np.array(coord, dtype=float))[0]
            pix = (pix * label_upscale_factor).astype(int)

            if not (0 <= pix[0] < new_w and 0 <= pix[1] < new_h):
                continue

            radius = max(10, int(7 * label_upscale_factor))
            circle_color = (0, 128, 255)  # orange border (BGR)
            fill_color = (255, 255, 255)  # white background
            border_thickness = max(2, label_upscale_factor)

            cv2.circle(overlay, tuple(pix), radius, fill_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, tuple(pix), radius, circle_color, border_thickness, lineType=cv2.LINE_AA)

            text_idx = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * label_upscale_factor
            font_thickness = max(1, label_upscale_factor)
            (tw, th), bl = cv2.getTextSize(text_idx, font, font_scale, font_thickness)
            text_org = (int(pix[0] - tw / 2), int(pix[1] + th / 2))
            cv2.putText(overlay, text_idx, text_org, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)

    t = time.time()
    agent_pixel_scaled = (np.array([map_size_px / 2, map_size_px / 2]) * label_upscale_factor).astype(int)

    fov_rad = np.deg2rad(fov_angle_deg)
    angles = np.linspace(-np.pi / 2 - fov_rad / 2, -np.pi / 2 + fov_rad / 2, 30)
    fov_range_px = fov_range / map_scale * label_upscale_factor
    sector_points = [(agent_pixel_scaled + (np.array([np.cos(a), np.sin(a)]) * fov_range_px)).astype(int) for a in
                     angles]
    sector_pts = np.array([agent_pixel_scaled] + sector_points, np.int32)

    overlay = top_down_map.copy()
    cv2.fillPoly(overlay, [sector_pts], (255, 255, 0), lineType=cv2.LINE_AA)
    top_down_map = cv2.addWeighted(overlay, 0.2, top_down_map, 0.8, 0)

    if subtasks:
        overlay = top_down_map.copy()

        new_h, new_w = overlay.shape[:2]

        for name, coord in subtasks.items():
            if coord is None:
                continue

            try:
                pix = transform_and_to_pixel(np.array(coord, dtype=float))[0]
            except Exception:
                continue
            pix = (pix * label_upscale_factor).astype(int)

            if not (0 <= pix[0] < new_w and 0 <= pix[1] < new_h):
                continue

            half = max(6, int(6 * label_upscale_factor))
            tl = (int(pix[0] - half), int(pix[1] - half))
            br = (int(pix[0] + half), int(pix[1] + half))

            tl = (max(0, tl[0]), max(0, tl[1]))
            br = (min(new_w - 1, br[0]), min(new_h - 1, br[1]))

            cv2.rectangle(overlay, tl, br, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.rectangle(overlay, tl, br, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            text = str(name)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * label_upscale_factor
            font_thickness = max(1, label_upscale_factor)
            (tw, th), bl = cv2.getTextSize(text, font, font_scale, font_thickness)

            tx = br[0] + int(6 * label_upscale_factor)
            ty = tl[1] - int(4 * label_upscale_factor)

            bx1 = max(0, tx - 3)
            by1 = max(0, ty - th - 3)
            bx2 = min(new_w - 1, tx + tw + 3)
            by2 = min(new_h - 1, ty + 3)

            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)

            text_org = (min(max(0, tx), new_w - 1 - tw), min(max(th, ty), new_h - 1))
            cv2.putText(overlay, text, text_org, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        top_down_map = cv2.addWeighted(overlay, 0.95, top_down_map, 0.05, 0)

    arrow_length = int(12 * label_upscale_factor * 1.2)  # shorter arrow, less sharp
    base_half = max(12, int(arrow_length * 0.6))  # wider base for blunter triangle

    triangle_pts = np.array([
        agent_pixel_scaled + np.array([0, -arrow_length]),  # apex
        agent_pixel_scaled + np.array([-base_half, int(arrow_length * 0.6)]),  # bottom-left
        agent_pixel_scaled + np.array([base_half, int(arrow_length * 0.6)])  # bottom-right
    ], dtype=np.int32)

    cv2.fillPoly(top_down_map, [triangle_pts], (0, 0, 255), lineType=cv2.LINE_AA)

    outline_thickness = max(3, 2 * label_upscale_factor)
    cv2.polylines(top_down_map, [triangle_pts], isClosed=True, color=(100, 100, 100),
                  thickness=outline_thickness, lineType=cv2.LINE_AA)

    timings['draw_agent'] = time.time() - t

    try:
        start_world = np.array([0.0, 0.0, 0.0])
        start_pixel = transform_and_to_pixel(start_world)[0]  # returns (x, y)
        start_pixel_scaled = (start_pixel * label_upscale_factor).astype(int)

        if 0 <= start_pixel_scaled[0] < new_w and 0 <= start_pixel_scaled[1] < new_h:

            cv2.circle(top_down_map, tuple(start_pixel_scaled), 7, (255, 0, 0), -1)  # solid blue
            cv2.circle(top_down_map, tuple(start_pixel_scaled), 9, (0, 0, 0), 2)  # black border

            text = "START"
            font_scale = 0.7
            thickness = 1
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_org = (start_pixel_scaled[0] + 10, start_pixel_scaled[1] + th // 2)

            bx1 = text_org[0] - 3
            by1 = text_org[1] - th - 3
            bx2 = text_org[0] + tw + 3
            by2 = text_org[1] + 3

            bx1 = max(bx1, 0);
            by1 = max(by1, 0);
            bx2 = min(bx2, new_w - 1);
            by2 = min(by2, new_h - 1)
            overlay = top_down_map.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.8, top_down_map, 0.2, 0, top_down_map)

            cv2.putText(top_down_map, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
                        cv2.LINE_AA)
    except Exception:

        pass

    crop_size_px = int(crop_size_m / map_scale * label_upscale_factor)
    agent_pixel_scaled = (np.array([map_size_px / 2, map_size_px / 2]) * label_upscale_factor).astype(int)
    half_crop = crop_size_px // 2
    y1 = max(agent_pixel_scaled[1] - half_crop, 0)
    y2 = min(agent_pixel_scaled[1] + half_crop, top_down_map.shape[0])
    x1 = max(agent_pixel_scaled[0] - half_crop, 0)
    x2 = min(agent_pixel_scaled[0] + half_crop, top_down_map.shape[1])
    top_down_map_cropped = top_down_map[y1:y2, x1:x2]

    top_down_map_cropped = draw_direction_markers(top_down_map_cropped)

    t_end = time.time()

    return top_down_map_cropped

def create_top_down_map_global(
    mapper: Instruct_Mapper,
    map_scale: float = 0.025,
    padding_px: int = 60,
    fov_angle_deg: float = 60.0,
    fov_range: float = 2.5,
    target_point=None,
    action_candidates=None,
    subtasks=None,
    enable_semantic: bool = True,
    enable_navigation: bool = True,
    show_all_labels: bool = False,
    label_volume_threshold: int = 100,
):
    """
    Global top-down view (not agent-centered).:
    - Auto-compute scene bounds and expand canvas.
    - enable_semantic: whether to render object point clouds and semantic labels.
    - enable_navigation: whether to render target / candidate waypoints / subtask nodes.
    """
    t_start = time.time()

    point_sets = []

    if hasattr(mapper, "navigable_pcd") and not mapper.navigable_pcd.is_empty():
        point_sets.append(mapper.navigable_pcd.point.positions.cpu().numpy()[:, :2])
    if hasattr(mapper, "obstacle_pcd") and not mapper.obstacle_pcd.is_empty():
        point_sets.append(mapper.obstacle_pcd.point.positions.cpu().numpy()[:, :2])
    if enable_semantic and hasattr(mapper, "object_entities"):
        for ent in mapper.object_entities:
            if ent['pcd'].point.positions.shape[0] > 0:
                point_sets.append(ent['pcd'].point.positions.cpu().numpy()[:, :2])
    if hasattr(mapper, "frontier_pcd") and not mapper.frontier_pcd.is_empty():
        point_sets.append(mapper.frontier_pcd.point.positions.cpu().numpy()[:, :2])
    if mapper.trajectory_position:
        traj_arr = np.array(mapper.trajectory_position)[:, :2]
        point_sets.append(traj_arr)
    if enable_navigation:
        if target_point is not None:
            point_sets.append(np.array(target_point)[None, :2])
        if action_candidates:
            for cand in action_candidates.values():
                coord = cand.get("target_point_world")
                if coord is not None:
                    point_sets.append(np.array(coord)[None, :2])
        if subtasks:
            for coord in subtasks.values():
                if coord is not None:
                    point_sets.append(np.array(coord)[None, :2])

    if not point_sets:

        map_size_px = 1024
        top_down_map = np.full((map_size_px, map_size_px, 3), 255, dtype=np.uint8)
        return top_down_map

    all_pts = np.vstack(point_sets)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)

    span = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

    base_size_px = int(span / map_scale) + 2 * padding_px

    base_size_px = max(base_size_px, 1024)

    width_px = int((max_xy[0] - min_xy[0]) / map_scale) + 2 * padding_px
    height_px = int((max_xy[1] - min_xy[1]) / map_scale) + 2 * padding_px
    width_px = max(width_px, 1024)
    height_px = max(height_px, 1024)

    top_down_map = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    map_h, map_w = top_down_map.shape[:2]
    offset = min_xy - padding_px * map_scale  # world-coord bottom-left offset

    def world_to_pixel(world_coords):
        arr = np.asarray(world_coords, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        px = (arr[:, :2] - offset) / map_scale
        return px.astype(int)

    pcd_resolution = 0.025
    dilation_radius = max(1, int(pcd_resolution / map_scale / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    if hasattr(mapper, "navigable_pcd") and not mapper.navigable_pcd.is_empty():
        nav_pts_world = mapper.navigable_pcd.point.positions.cpu().numpy()
        nav_px = world_to_pixel(nav_pts_world)
        valid = (nav_px[:, 0] >= 0) & (nav_px[:, 0] < map_w) & (nav_px[:, 1] >= 0) & (nav_px[:, 1] < map_h)
        if np.any(valid):
            mask = np.zeros((map_h, map_w), dtype=np.uint8)
            mask[nav_px[valid, 1], nav_px[valid, 0]] = 255
            mask = cv2.dilate(mask, kernel)
            top_down_map[mask == 255] = (200, 200, 200)

    if enable_navigation:
        try:
            wps = mapper.get_candidate_waypoints(min_distance=0.3, max_distance=1.5, waypoint_grid_resolution=0.5)
            if wps is not None:

                if hasattr(wps, "detach"):
                    wps = wps.detach().cpu().numpy()
                else:
                    wps = np.asarray(wps)
                if wps.size > 0:
                    if wps.ndim == 1:
                        wps = wps.reshape(1, -1)

                    wp_xy = wps[:, :2]
                    wp_px = world_to_pixel(wp_xy)
                    valid = (wp_px[:, 0] >= 0) & (wp_px[:, 0] < map_w) & (wp_px[:, 1] >= 0) & (wp_px[:, 1] < map_h)
                    if np.any(valid):
                        overlay = top_down_map.copy()
                        for p in wp_px[valid]:

                            cv2.circle(overlay, tuple(p), 2, (0, 165, 255), -1, cv2.LINE_AA)
                        top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)
        except Exception:

            pass

    obstacle_mask_bool = np.zeros((map_h, map_w), dtype=bool)
    if hasattr(mapper, "obstacle_pcd") and not mapper.obstacle_pcd.is_empty():
        obs_pts_world = mapper.obstacle_pcd.point.positions.cpu().numpy()
        obs_px = world_to_pixel(obs_pts_world)
        valid = (obs_px[:, 0] >= 0) & (obs_px[:, 0] < map_w) & (obs_px[:, 1] >= 0) & (obs_px[:, 1] < map_h)
        if np.any(valid):
            mask = np.zeros((map_h, map_w), dtype=np.uint8)
            mask[obs_px[valid, 1], obs_px[valid, 0]] = 255
            mask = cv2.dilate(mask, kernel)
            top_down_map[mask == 255] = (50, 50, 50)
            obstacle_mask_bool = (mask == 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(top_down_map, contours, -1, (0, 0, 0), thickness=4)

    if hasattr(mapper, "frontier_pcd") and not mapper.frontier_pcd.is_empty():
        fr_px = world_to_pixel(mapper.frontier_pcd.point.positions.cpu().numpy())
        valid = (fr_px[:, 0] >= 0) & (fr_px[:, 0] < map_w) & (fr_px[:, 1] >= 0) & (fr_px[:, 1] < map_h)
        for p in fr_px[valid]:
            cv2.circle(top_down_map, tuple(p), 1, (0, 255, 255), -1)

    if enable_semantic and hasattr(mapper, "object_entities"):
        for ent in mapper.object_entities:
            if ent['pcd'].point.positions.shape[0] == 0:
                continue
            obj_px = world_to_pixel(ent['pcd'].point.positions.cpu().numpy())
            valid = (obj_px[:, 0] >= 0) & (obj_px[:, 0] < map_w) & (obj_px[:, 1] >= 0) & (obj_px[:, 1] < map_h)
            if not np.any(valid):
                continue
            obj_valid_px = obj_px[valid]

            inter_mask = obstacle_mask_bool[obj_valid_px[:, 1], obj_valid_px[:, 0]]
            if np.any(inter_mask):
                final_px = obj_valid_px[inter_mask]
                color_arr = ent['pcd'].point.colors.cpu().numpy()[valid][inter_mask]
            else:
                final_px = obj_valid_px
                color_arr = ent['pcd'].point.colors.cpu().numpy()[valid]
            if final_px.shape[0] == 0:
                continue

            obj_color = (color_arr[0] * 255)[::-1].astype(np.uint8)
            mask_img = np.zeros((map_h, map_w), dtype=np.uint8)
            mask_img[final_px[:, 1], final_px[:, 0]] = 255
            fill_kernel = np.ones((5, 5), np.uint8)
            filled = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, fill_kernel, iterations=2)
            top_down_map[filled == 255] = obj_color

    if False and enable_semantic and hasattr(mapper, "object_entities"):
        drawn_labels = []
        merge_dist_px = 40
        for ent in sorted(mapper.object_entities, key=lambda e: e['class']):
            if ent['pcd'].point.positions.shape[0] == 0:
                continue
            if (not show_all_labels) and ent['pcd'].point.positions.shape[0] < label_volume_threshold:
                continue
            center_px = world_to_pixel(ent['center'])[0]
            if not (0 <= center_px[0] < map_w and 0 <= center_px[1] < map_h):
                continue
            cname = ent['class_name']
            merged = False
            for i, info in enumerate(drawn_labels):
                if info['class'] == cname and np.linalg.norm(center_px - info['center']) < merge_dist_px:
                    drawn_labels[i]['center'] = (info['center'] + center_px) / 2
                    merged = True
                    break
            if not merged:
                drawn_labels.append({'class': cname, 'center': center_px})

        for info in drawn_labels:
            cp = info['center'].astype(int)
            font_scale = 0.6
            thickness = 1
            (tw, th), bl = cv2.getTextSize(info['class'], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tl = (cp[0] - tw // 2 - 3, cp[1] - th - 4)
            br = (cp[0] + tw // 2 + 3, cp[1] + bl + 2)
            tl = (max(0, tl[0]), max(0, tl[1]))
            br = (min(map_w - 1, br[0]), min(map_h - 1, br[1]))
            overlay = top_down_map.copy()
            cv2.rectangle(overlay, tl, br, (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.85, top_down_map, 0.15, 0, top_down_map)
            cv2.rectangle(top_down_map, tl, br, (0, 0, 0), 1)
            text_org = (tl[0] + 3, br[1] - bl)
            cv2.putText(top_down_map, info['class'], text_org, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    if enable_navigation and mapper.trajectory_position and len(mapper.trajectory_position) > 1:
        traj_px = world_to_pixel(np.array(mapper.trajectory_position[-200:])[:, :2]).astype(int)
        valid = (traj_px[:, 0] >= 0) & (traj_px[:, 0] < map_w) & (traj_px[:, 1] >= 0) & (traj_px[:, 1] < map_h)
        traj_px = traj_px[valid]
        if traj_px.shape[0] > 1:
            overlay = top_down_map.copy()
            cv2.polylines(overlay, [traj_px], False, (0, 0, 255), 2)
            top_down_map = cv2.addWeighted(overlay, 0.6, top_down_map, 0.4, 0)

    if enable_navigation:

        # if target_point is not None:
        #     tp = world_to_pixel(np.array(target_point))[0]
        #     if 0 <= tp[0] < map_w and 0 <= tp[1] < map_h:
        #         cv2.circle(top_down_map, tuple(tp), 9, (0, 255, 0), -1)
        #         cv2.circle(top_down_map, tuple(tp), 9, (0, 0, 0), 2)

        if action_candidates:
            overlay = top_down_map.copy()
            for text, cand in action_candidates.items():
                if cand.get("type") != "waypoint":
                    continue
                coord = cand.get("target_point_world")
                if coord is None:
                    continue
                pix = world_to_pixel(np.array(coord))[0]
                if not (0 <= pix[0] < map_w and 0 <= pix[1] < map_h):
                    continue

                try:
                    idx = int(text.split()[-1])
                except Exception:
                    idx = None
                radius = 11
                cv2.circle(overlay, tuple(pix), radius, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, tuple(pix), radius, (0, 128, 255), 2, cv2.LINE_AA)
                if idx is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), bl = cv2.getTextSize(str(idx), font, 0.6, 2)
                    org = (int(pix[0] - tw / 2), int(pix[1] + th / 2))
                    cv2.putText(overlay, str(idx), org, font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)

        if subtasks:
            overlay = top_down_map.copy()
            for name, coord in subtasks.items():
                if coord is None:
                    continue
                pix = world_to_pixel(np.array(coord))[0]
                if not (0 <= pix[0] < map_w and 0 <= pix[1] < map_h):
                    continue

                half = 10
                tl = (pix[0] - half, pix[1] - half)
                br = (pix[0] + half, pix[1] + half)
                tl = (max(0, tl[0]), max(0, tl[1]))
                br = (min(map_w - 1, br[0]), min(map_h - 1, br[1]))

                cv2.rectangle(overlay, tl, br, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.rectangle(overlay, tl, br, (0, 0, 0), 1, cv2.LINE_AA)

                text = str(name)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45  # fit 16x16 block
                thickness = 1
                (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)

                cx = (tl[0] + br[0]) // 2
                cy = (tl[1] + br[1]) // 2

                text_org = (int(cx - tw / 2), int(cy + th / 2) - 1)

                cv2.putText(overlay, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            top_down_map = cv2.addWeighted(overlay, 0.95, top_down_map, 0.05, 0)

    agent_pos = getattr(mapper, "current_position", np.array([0.0, 0.0, 0.0]))
    agent_rot = getattr(mapper, "current_rotation", np.eye(3))
    agent_px = world_to_pixel(agent_pos)[0]
    if 0 <= agent_px[0] < map_w and 0 <= agent_px[1] < map_h:
        forward_vec_3d = agent_rot @ np.array([0, 0, -1])
        forward_2d = forward_vec_3d[:2]
        if np.linalg.norm(forward_2d) < 1e-6:
            forward_2d = np.array([1.0, 0.0])
        forward_2d /= np.linalg.norm(forward_2d)
        angle = np.arctan2(forward_2d[1], forward_2d[0])

        # fov_rad = np.deg2rad(fov_angle_deg)
        # num_pts = 40
        # angles = np.linspace(angle - fov_rad / 2, angle + fov_rad / 2, num_pts)
        # fov_range_px = fov_range / map_scale
        # sector_pts = [agent_px]
        # for a in angles:
        #     end_world = agent_pos[:2] + np.array([np.cos(a), np.sin(a)]) * fov_range
        #     sector_pts.append(world_to_pixel(end_world)[0])
        # overlay = top_down_map.copy()
        # cv2.fillPoly(overlay, [np.array(sector_pts, dtype=np.int32)], (255, 255, 0))
        # top_down_map = cv2.addWeighted(overlay, 0.2, top_down_map, 0.8, 0)

        length = 30
        base_half = 10
        tip = agent_px + (forward_2d * length).astype(int)
        left_dir = np.array([[0, -1], [1, 0]]) @ forward_2d
        right_dir = -left_dir
        p_left = agent_px + (forward_2d * (length * 0.2) + left_dir * base_half).astype(int)
        p_right = agent_px + (forward_2d * (length * 0.2) + right_dir * base_half).astype(int)
        tri = np.array([tip, p_left, p_right], dtype=np.int32)
        cv2.fillPoly(top_down_map, [tri], (0, 0, 255))
        cv2.polylines(top_down_map, [tri], True, (50, 50, 50), 2)

    start_px = world_to_pixel(np.array([0.0, 0.0]))[0]
    if enable_navigation and 0 <= start_px[0] < map_w and 0 <= start_px[1] < map_h:
        cv2.circle(top_down_map, tuple(start_px), 7, (255, 0, 0), -1)
        cv2.circle(top_down_map, tuple(start_px), 9, (0, 0, 0), 2)
        (tw, th), bl = cv2.getTextSize("START", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        org = (start_px[0] + 10, start_px[1] + th // 2)
        bx1, by1 = org[0] - 3, org[1] - th - 3
        bx2, by2 = org[0] + tw + 3, org[1] + 3
        bx1 = max(bx1, 0); by1 = max(by1, 0)
        bx2 = min(bx2, map_w - 1); by2 = min(by2, map_h - 1)
        overlay = top_down_map.copy()
        # cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, top_down_map, 0.2, 0, top_down_map)
        # cv2.putText(top_down_map, "START", org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.cvtColor(top_down_map, cv2.COLOR_BGR2RGB)


class Instruction:
    """
    Class encapsulating task instructions, decomposition, and subtask management.
    Each subtask has a final_coord attribute, recorded only when marked complete.
    """
    def __init__(self, full_instruction: str, sub_instructions: list = None):
        """
        Initialize Instruction object.
        Contains the full task instruction and an optional list of subtask instructions.
        :param full_instruction: full task instruction text.
        :param sub_instructions: decomposed subtask instruction list.
        """
        self.full_instruction = full_instruction
        self.sub_instruction_dict = {}

        if sub_instructions is not None:
            for idx, sub_inst in enumerate(sub_instructions):
                key = f'P{idx + 1}'
                inst_body = {}
                inst_body['instruction'] = sub_inst
                inst_body['completed'] = False
                inst_body['plan'] = ""
                inst_body['record'] = []
                inst_body['final_coord'] = None
                self.sub_instruction_dict[key] = inst_body

        self.sub_instruction_keys = list(self.sub_instruction_dict.keys())
        self.current_subtask_index = 0
        self.current_step_count = 0
        self.num_subtasks = len(self.sub_instruction_keys)

    def get_current_subtask_key(self):
        """Get the current subtask key."""
        if self.current_subtask_index < self.num_subtasks:
            return self.sub_instruction_keys[self.current_subtask_index]
        return None

    def is_last_subtask(self):
        """Check if the current subtask is the last one."""
        return self.current_subtask_index == self.num_subtasks - 1

    def get_current_subtask(self):
        """Get current subtask instruction (key + dict)."""
        key = self.get_current_subtask_key()
        if key:
            return key, self.sub_instruction_dict[key]
        return None, None

    def update_plan_for_current_subtask(self, plan: str):
        """Update the plan text for the current subtask."""
        key = self.get_current_subtask_key()
        if key:
            self.sub_instruction_dict[key]['plan'] = plan

    def add_record_to_current_subtask(self, record: dict):
        """Append an execution record to the current subtask."""
        key = self.get_current_subtask_key()
        if key:
            self.sub_instruction_dict[key]['record'].append(record)

    def reset_subtask(self, key: str):
        """Reset the subtask for the given key to incomplete, clearing plan/records/coords/rotation."""
        key_upper = key.upper()
        if key_upper in self.sub_instruction_dict:
            self.sub_instruction_dict[key_upper]['completed'] = False
            self.sub_instruction_dict[key_upper]['plan'] = ""
            self.sub_instruction_dict[key_upper]['record'] = []
            self.sub_instruction_dict[key_upper]['final_coord'] = None

            self.sub_instruction_dict[key_upper]['final_rotation'] = None
        self.current_step_count = 0

    def get_subtask_by_key(self, key: str):
        """Get subtask instruction by key (case-insensitive)."""
        key_upper = key.upper()
        return self.sub_instruction_dict.get(key_upper, None)

    def get_all_subtasks_str(self):
        """
        Get JSON string representation of all subtasks (only current has record + final_coord).
        """
        result = []
        for idx, key in enumerate(self.sub_instruction_keys):
            subtask = self.sub_instruction_dict[key]

            if subtask['completed']:
                status = "completed"
            elif idx == self.current_subtask_index:
                status = "ongoing(current task)"
            else:
                status = "queuing"

            if idx == self.current_subtask_index:
                record_dict = {}
                for i, rec in enumerate(subtask['record']):
                    record_key = f"{key.lower()}.{i + 1}"
                    record_dict[record_key] = rec
            else:
                record_dict = {}

            task_info = {
                "task_name": key.lower(),
                "instruction": subtask['instruction'],
                "status": status,
                "record": record_dict,
                "your_last_plan": subtask['plan'],
            }
            result.append(task_info)

        all_tasks = json.dumps(result, ensure_ascii=False, indent=0)
        cur_task = f"\n\nCurrent task: {self.get_current_subtask_key()}: {self.get_current_subtask()[1]['instruction']}"
        return f"Full instruction: {self.full_instruction}\n" + all_tasks + cur_task

    def mark_current_subtask_completed(self, coord=None, rotation=None):
        """
        Mark the current subtask as completed and advance to the next.
        coord: final coordinate (x, y, z).
        rotation: final orientation quaternion (supports mn.Quaternion / [w,x,y,z] / objects with w,x,y,z).
        """
        key = self.get_current_subtask_key()
        if key:
            self.sub_instruction_dict[key]['completed'] = True

            if coord is not None:
                try:
                    self.sub_instruction_dict[key]['final_coord'] = [
                        float(coord[0]), float(coord[1]), float(coord[2])
                    ]
                except Exception:
                    self.sub_instruction_dict[key]['final_coord'] = None

            if rotation is not None:
                self.sub_instruction_dict[key]['final_rotation'] = rotation

            if self.current_subtask_index < self.num_subtasks - 1:
                self.current_subtask_index += 1
            self.current_step_count = 0


    def is_all_completed(self):
        """Check if all subtasks are completed."""
        return all(subtask['completed'] for subtask in self.sub_instruction_dict.values())

    def roll_back_to_specific_subtask(self, key: str):
        """Roll back to the subtask identified by key."""
        key_upper = key.upper()
        if key_upper in self.sub_instruction_keys:
            target_index = self.sub_instruction_keys.index(key_upper)
            self.current_subtask_index = target_index

            for i in range(target_index, self.num_subtasks):
                reset_key = self.sub_instruction_keys[i]
                self.reset_subtask(reset_key)

    def get_subtasks_key_coord(self):
        """Get a dict of all completed subtask keys with their final coordinates."""
        result = {}
        for key in self.sub_instruction_keys:
            subtask = self.sub_instruction_dict[key]
            if subtask['completed'] and subtask['final_coord'] is not None:
                result[key] = subtask['final_coord']
        return result

    def get_subtask_pos_rot_by_key(self, key: str):
        """Get the final coordinate and rotation for the given subtask key. Returns (coord, rotation) tuple."""
        key_upper = key.upper()
        subtask = self.sub_instruction_dict.get(key_upper, None)
        if subtask and subtask['completed']:
            coord = subtask.get('final_coord', None)
            rotation = subtask.get('final_rotation', None)
            return coord, rotation
        return None, None

import json
import re

def robust_json_parse(json_str: str):
    """
    Parse a JSON string that may contain comments or common formatting errors; returns dict or None.
    """

    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    json_str = json_str.strip()

    json_str = re.sub(r'^```json', '', json_str)
    json_str = re.sub(r'^```', '', json_str)
    json_str = re.sub(r'```$', '', json_str)

    try:
        return json.loads(json_str)
    except Exception as e:
        pass
        return None

class PathPlannerAgent(ABC):
    """
    An Agent whose decision logic is:
    1. Generate waypoints in current view plus basic actions (L, R, 0).
    2. Let the VLM choose a target waypoint or action.
    3. Use A* to plan a path to that point.
    4. Decompose the path into actions and execute step-by-step, updating the map each step.
    """

    def __init__(self, sim_wrapper: SimWrapper, config: dict, instruction: str, init_position=None, init_rotation=None, mode='objnav'):
        self.sim_wrapper = sim_wrapper
        self.config = config
        self.forward_step = config.get('forward_step', 1.0)
        self.turn_angle_rad = np.deg2rad(config.get('turn_angle_deg', 90))
        self.instruction = instruction
        self.curr_obs = None
        self.last_action = PolarAction.null
        self.last_actions = {}
        self._recent_positions = []
        self.resolution = (config['camera']['height'], config['camera']['width'])
        self.focal_length = calculate_focal_length(config['camera']['fov'], self.resolution[1])
        self.image_edge_threshold = 0.04
        self.traveled_distance = 0.0
        self.prev_agent_position = None
        self.vlm_responses = []
        self.mode = mode
        self.error = False

        sub_insts = self.decompose_instruction(instruction)
        # sub_insts = [instruction]

        self.instruction_obj = Instruction(full_instruction=instruction, sub_instructions=sub_insts)

        if init_position is not None and init_rotation is not None:
            self.sim_wrapper.set_initial_state(init_position, init_rotation)

        camera_params = config['camera']
        cam_intrinsics = np.array([
            [camera_params['fx'], 0, camera_params['width'] / 2],
            [0, camera_params['fy'], camera_params['height'] / 2],
            [0, 0, 1]
        ])

        _map = config.get('map', {})
        self.mapper = Instruct_Mapper(camera_intrinsic=cam_intrinsics,
                                      grid_resolution=_map.get('grid_resolution', 0.05),
                                      floor_height=_map.get('floor_height', -1.2),
                                      ceiling_height=_map.get('ceiling_height', 0.6),
                                      pcd_resolution=_map.get('pcd_resolution', 0.025),
                                      resolution=(config['camera']['height'], config['camera']['width']))

        initial_state = self.sim_wrapper.sim.get_agent(0).get_state()
        self.mapper.reset(initial_state.position, initial_state.rotation)

        obs = sim_wrapper.step(PolarAction(r=0, theta=0))
        self.mapper.reset(obs['agent_state'].position, obs['agent_state'].rotation)
        self.instances = self.update_map(obs)
        self.curr_obs = obs
        self.turn_cooldown = 0
        self.turn_count = 0
        self.current_plan = None
        self.last_vlm_response = "Executing initial action."
        self.target_world_position = [0, 0, 0]
        self.img_buffer = []
        self.min_move_distance = 0.4
        self.max_move_distance = 3

        self.init_pos = self.mapper.current_position - [0, 0, 1.2]
        self.init_rot = self.mapper.current_rotation

        self.first_stop = False
        self.second_stop = False

        self.rolling_back = False

        stair_words = ['stair', 'step']
        if any(word in instruction.lower() for word in stair_words):
            pass
            self.mapper.enable_stair_detection = True
        else:
            self.mapper.enable_stair_detection = False

    def _stop(self):
        """Execute stop action and update sub-goal state."""
        if not self.first_stop:
            pass
            self.first_stop = True
            return PolarAction.stop
        elif not self.second_stop:
            pass
            self.second_stop = True
            return PolarAction.stop
        else:
            pass
            return PolarAction.null()

    def update_map(self, obs: dict):
        """Update map using four-direction observations (batched multiview)."""

        views = []

        if 'left' in obs and 'agent_state' in obs['left']:
            s = obs['left']['agent_state']
            views.append({
                'rgb': obs['left']['color_sensor'],
                'depth': obs['left']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        if 'right' in obs and 'agent_state' in obs['right']:
            s = obs['right']['agent_state']
            views.append({
                'rgb': obs['right']['color_sensor'],
                'depth': obs['right']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        if 'back' in obs and 'agent_state' in obs['back']:
            s = obs['back']['agent_state']
            views.append({
                'rgb': obs['back']['color_sensor'],
                'depth': obs['back']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        agent_state = obs['agent_state']
        # Primary (front) view — always last so primary_index defaults to it
        views.append({
            'rgb': obs['color_sensor'],
            'depth': obs['depth_sensor'],
            'position': np.array(agent_state.position),
            'rotation': agent_state.sensor_states['color_sensor'].rotation,
        })

        instances = self.mapper.update_multiview(views)

        return instances

        return instances

    def _world_to_pixel_coords(self, waypoints_world, agent_position, agent_rotation, camera_intrinsic):
        """
        Convert world-coordinate waypoints to image pixel coordinates.
        :param waypoints_world:
        :return:
        """
        if waypoints_world.shape[0] == 0:
            return np.array([]).reshape(0, 2), np.array([], dtype=int), np.array([], dtype=float)

        relative_points = waypoints_world - agent_position
        if hasattr(agent_rotation, 'as_rotation_matrix'):
            rotation_matrix = agent_rotation.as_rotation_matrix()
        elif hasattr(agent_rotation, 'shape') and len(agent_rotation.shape) == 1 and agent_rotation.shape[0] == 4:
            rotation_matrix = R.from_quat(agent_rotation).as_matrix()
        else:
            rotation_matrix = agent_rotation
        rotation_matrix_inv = rotation_matrix.T
        local_points = np.dot(relative_points, rotation_matrix_inv.T)

        camera_points = np.copy(local_points)
        camera_points[:, 1] = -local_points[:, 1]
        camera_points[:, 2] = -local_points[:, 2]

        original_indices = np.arange(waypoints_world.shape[0])
        valid_mask = camera_points[:, 2] > 0.1
        if not np.any(valid_mask):
            return np.array([]).reshape(0, 2), np.array([], dtype=int), np.array([], dtype=float)

        valid_points = camera_points[valid_mask]
        valid_original_indices = original_indices[valid_mask]
        camera_depths = valid_points[:, 2].astype(float)

        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
        pixel_x = (valid_points[:, 0] * fx / valid_points[:, 2]) + cx
        pixel_y = (valid_points[:, 1] * fy / valid_points[:, 2]) + cy
        pixel_coords = np.column_stack([pixel_x, pixel_y])

        return pixel_coords, valid_original_indices, camera_depths

    def _preprocessing_module(self, obs, use_far_waypoints: bool = False):
        """
        Preprocessing: generate waypoints in current view, project onto image, including turn-left/right actions.
        Also mark visited candidates (<1m from trajectory) in red using trajectory_position history.
        """
        import numpy as np
        import cv2
        from scipy.spatial import cKDTree

        image = obs['color_sensor'].copy()
        height, width, _ = image.shape

        depth_img = obs.get('depth_sensor', None)
        depth_h = depth_w = None
        sx = sy = 1.0
        if depth_img is not None:
            if depth_img.ndim == 3 and depth_img.shape[2] == 1:
                depth_img = depth_img[:, :, 0]
            depth_h, depth_w = depth_img.shape[:2]
            sx = depth_w / float(width)
            sy = depth_h / float(height)

        actions = {}
        action_idx = 1

        waypoints_world_near, waypoints_relative_near = self.mapper.get_current_view_candidate_waypoints(
            waypoint_grid_resolution=1.0, min_distance=0.3, max_distance=1.5, merge_distance=0.4
        )

        if use_far_waypoints:
            waypoints_world_far, waypoints_relative_far = self.mapper.get_current_view_candidate_waypoints(
                waypoint_grid_resolution=2.0, min_distance=0.3, max_distance=1.5, merge_distance=1.5
            )
            agent_pos = self.mapper.current_position
            if waypoints_world_far.shape[0] > 0:
                distances_far = np.linalg.norm(waypoints_world_far - agent_pos, axis=1)
                far_mask = (distances_far > self.max_move_distance) & (distances_far < 8.0)
                waypoints_world_far = waypoints_world_far[far_mask]
                waypoints_relative_far = waypoints_relative_far[far_mask]
            else:
                waypoints_world_far = np.empty((0, 3))
                waypoints_relative_far = np.empty((0, 3))
        else:
            waypoints_world_far = np.empty((0, 3))
            waypoints_relative_far = np.empty((0, 3))

        if waypoints_world_far.shape[0] > 0:
            waypoints_world = np.vstack([waypoints_world_near, waypoints_world_far])
            waypoints_relative = np.vstack([waypoints_relative_near, waypoints_relative_far])
        else:
            waypoints_world = waypoints_world_near
            waypoints_relative = waypoints_relative_near

        visited_radius = 0.8
        if waypoints_world.shape[0] > 0:
            traj = np.array(getattr(self.mapper, 'trajectory_position', []), dtype=float)[:-1] - np.array([0, 0, 1.2])
            if traj.size > 0:

                traj_xz = traj[:, [0, 1]]
                wpts_xz = waypoints_world[:, [0, 1]]
                tree = cKDTree(traj_xz)

                dists, _ = tree.query(wpts_xz, k=1)
                visited_mask_all = dists < visited_radius
            else:
                visited_mask_all = np.zeros((waypoints_world.shape[0],), dtype=bool)
        else:
            visited_mask_all = np.zeros((0,), dtype=bool)

        pixel_coords, original_indices, cam_depths = self._world_to_pixel_coords(
            waypoints_world,
            self.mapper.current_position,
            self.mapper.current_rotation,
            self.mapper.camera_intrinsic
        )

        far_count = int(waypoints_world_far.shape[0]) if use_far_waypoints else 0


        # Build the A* graph buffer (mapper.waypoints) from the global navigable
        # map so plan_path_to_target() below has nodes to route through. Without
        # this the buffer stays empty and every path query returns [] (no_path),
        # leaving the agent with no selectable waypoint.
        self.mapper.get_candidate_waypoints(
            waypoint_grid_resolution=0.5, min_distance=0.3, max_distance=2.5
        )

        skip_by_edge = 0
        skip_by_distance = 0
        skip_by_path = 0
        skip_by_occlusion = 0
        print(f"waypoints: {waypoints_world.shape[0]}")

        if pixel_coords.shape[0] > 0:
            for i in range(pixel_coords.shape[0]):
                x_pixel, y_pixel = int(pixel_coords[i, 0]), int(pixel_coords[i, 1])
                original_idx = int(original_indices[i])

                edge_margin_x = width * self.image_edge_threshold
                edge_margin_y = height * self.image_edge_threshold
                if not (edge_margin_x < x_pixel < width - edge_margin_x and
                        edge_margin_y < y_pixel < height - edge_margin_y):
                    skip_by_edge += 1
                    continue

                if depth_img is not None and cam_depths.shape[0] > i:
                    dx = int(np.clip(round(x_pixel * sx), 0, (depth_w - 1)))
                    dy = int(np.clip(round(y_pixel * sy), 0, (depth_h - 1)))
                    observed = float(depth_img[dy, dx])
                    if np.isfinite(observed) and observed > 0:
                        tol = max(0.12, 0.03 * observed)
                        if cam_depths[i] > observed + tol:
                            skip_by_occlusion += 1
                            continue

                wp_world = waypoints_world[original_idx]
                wp_relative = waypoints_relative[original_idx] + [0, 1.2, 0]

                wp_world_for_planning = np.copy(wp_world)
                distance = np.linalg.norm(wp_relative)

                if distance <= self.max_move_distance:
                    if distance < self.min_move_distance:
                        skip_by_distance += 1
                        continue
                else:
                    if not use_far_waypoints or distance >= 10.0:
                        skip_by_distance += 1
                        continue

                path = self.mapper.plan_path_to_target(wp_world_for_planning)
                if not path or len(path) == 0:
                    skip_by_path += 1
                    continue

                is_visited = bool(visited_mask_all[original_idx])

                min_radius, max_radius = 6, 35
                min_font, max_font = 0.4, 1.1
                norm_dist = np.clip(distance / 3, 0, 1)
                font_scale = max_font - (max_font - min_font) * norm_dist

                text = str(action_idx)
                (ref_text_width, ref_text_height), _ = cv2.getTextSize("99", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_diag = int(np.sqrt(ref_text_width ** 2 + ref_text_height ** 2))
                radius = max(int(text_diag / 2 + 2), min_radius)

                overlay = image.copy()

                circle_color = (255, 200, 100) if distance > self.max_move_distance else (255, 255, 255)
                if is_visited:
                    circle_color = (255, 100, 100)

                cv2.circle(overlay, (x_pixel, y_pixel), radius, circle_color, -1)
                cv2.circle(overlay, (x_pixel, y_pixel), radius, (0, 0, 0), 2)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                cv2.putText(image, text, (x_pixel - text_width // 2, y_pixel + text_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

                actions[text] = {
                    'type': 'waypoint',
                    'target_point_world': waypoints_world[original_idx],
                    'target_point_relative': waypoints_relative[original_idx],
                    'visited': is_visited
                }
                action_idx += 1

        print(
            f"skipped waypoints: edge {skip_by_edge},  distance {skip_by_distance},  no_path {skip_by_path},  occlusion {skip_by_occlusion}")

        if self.first_stop:
            try:
                num_points = 50
                radius = 1.0
                thetas = np.linspace(0, 2 * np.pi, num_points)
                agent_pos = self.mapper.current_position
                floor_z = agent_pos[2] - 1.2

                circle_x = agent_pos[0] + radius * np.cos(thetas)
                circle_y = agent_pos[1] + radius * np.sin(thetas)
                circle_z = np.full_like(circle_x, floor_z)
                circle_world_points = np.vstack([circle_x, circle_y, circle_z]).T

                pixel_coords_circle, _, _ = self._world_to_pixel_coords(
                    circle_world_points,
                    self.mapper.current_position,
                    self.mapper.current_rotation,
                    self.mapper.camera_intrinsic
                )

                if pixel_coords_circle.shape[0] > 1:
                    pts = pixel_coords_circle.astype(np.int32).reshape((-1, 1, 2))
                    overlay = image.copy()
                    cv2.polylines(
                        overlay,
                        [pts],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                    scale_factor = image.shape[0] / 1080.0
                    top_idx = np.argmin(pixel_coords_circle[:, 1])
                    top_point = pixel_coords_circle[top_idx]
                    text = "1m"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2 * scale_factor
                    thickness = max(2, int(2 * scale_factor))
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

                    tx = int(top_point[0] - tw / 2)
                    ty = int(top_point[1] - 2 * scale_factor - th)

                    tx = max(2, min(tx, image.shape[1] - tw - 2))
                    ty = max(th + 2, min(ty, image.shape[0] - 2))

                    cv2.putText(image, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    cv2.putText(image, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            except Exception as e:
                pass

        action_keys = ['L', 'R', 'B']
        positions = [(50, height // 2), (width - 50, height // 2), (width // 2, 50)]
        angles = [self.turn_angle_rad, -self.turn_angle_rad, np.pi]
        for key, pos, angle in zip(action_keys, positions, angles):
            overlay = image.copy()
            cv2.circle(overlay, pos, 30, (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.putText(image, key, (pos[0] - 15, pos[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            actions[key] = {'type': 'turn', 'angle': angle}

        subtask_points = self.instruction_obj.get_subtasks_key_coord()
        try:
            if subtask_points:
                keys, pts = [], []
                for k, coord in subtask_points.items():
                    if coord is None or len(coord) < 3:
                        continue
                    pts.append([float(coord[0]), float(coord[1]), float(coord[2])])
                    keys.append(k)

                if len(pts) > 0:
                    pts_np = np.array(pts, dtype=float)
                    pixel_coords_st, valid_idx_st, _ = self._world_to_pixel_coords(
                        pts_np,
                        self.mapper.current_position,
                        self.mapper.current_rotation,
                        self.mapper.camera_intrinsic
                    )

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale_factor = image.shape[0] / 1080.0
                    for i, px in enumerate(pixel_coords_st):
                        key = keys[int(valid_idx_st[i])]
                        x, y = int(px[0]), int(px[1])
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue

                        half = max(6, int(8 * scale_factor))
                        tl = (max(0, x - half), max(0, y - half))
                        br = (min(width - 1, x + half), min(height - 1, y + half))
                        cv2.rectangle(image, tl, br, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                        cv2.rectangle(image, tl, br, (0, 0, 0), 1, lineType=cv2.LINE_AA)

                        label = key
                        font_scale = max(0.5, 0.8 * scale_factor)
                        thickness = max(1, int(2 * scale_factor))
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        tx = br[0] + int(6 * scale_factor)
                        ty = tl[1] - int(4 * scale_factor)

                        bx1 = max(0, tx - 3)
                        by1 = max(0, ty - th - 3)
                        bx2 = min(width - 1, tx + tw + 3)
                        by2 = min(height - 1, ty + 3)
                        if bx2 > bx1 and by2 > by1:
                            cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
                            cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 0, 0), 1)

                        text_org = (min(max(0, tx), width - 1 - tw), min(max(th, ty), height - 1))
                        cv2.putText(image, label, text_org, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        except Exception as e:
            pass

        return image, actions

    def decide_waypoint(self, is_stuck=False, is_first_step=False):
        """
        Decide the current target waypoint or action.
        This function analyzes the current state (map + 4 views) via VLM, selects a waypoint or action,
        then returns the action details for subsequent planning and execution.
        :return:
                 or {'type': 'turn', 'angle': ...}. If VLM decides to stop, returns {'type': 'stop'}.
                 Returns None if all attempts fail.
        """
        obs = self.curr_obs

        labeled_image_np, actions = self._preprocessing_module(obs)
        self.last_actions = actions

        img = labeled_image_np.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        label = 'Front View'
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        text_x = img.shape[1] - text_size[0] - 20
        text_y = 50

        overlay = img.copy()
        cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (255, 255, 255), -1)
        cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        pil_labeled_rgb_image = Image.fromarray(img[:, :, :3], 'RGB')

        view_images = []
        for direction, label in [('left', 'Left View'), ('right', 'Right View'), ('back', 'Back View')]:
            if direction in obs and 'color_sensor' in obs[direction]:
                img = obs[direction]['color_sensor'].copy()

                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = img.shape[1] - text_size[0] - 20
                text_y = 50

                overlay = img.copy()
                cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10),
                              (255, 255, 255), -1)
                cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                view_images.append(Image.fromarray(img[:, :, :3], 'RGB'))

        top_down_map_np = create_top_down_map_centered(self.mapper, self.config['camera']['fov'],
                                                       self.target_world_position, action_candidates=self.last_actions, subtasks=self.instruction_obj.get_subtasks_key_coord())
        top_down_map_rgb = cv2.cvtColor(top_down_map_np, cv2.COLOR_BGR2RGB)
        pil_map_image = Image.fromarray(top_down_map_rgb)

        active_instr = self.instruction
        prompt = self.generate_prompt(active_instr, is_stuck, is_first_step)

        if len(actions) == 3:
            prompt += "\nNo available waypoints detected in current view. Please choose to turn left (L), turn right (R), or turn around (B) to explore more."

        max_retry = 3
        for attempt in range(max_retry):
            try:
                if attempt > 0:
                    prompt += "\nNote: Please carefully choose a valid action from the provided options and ensure your output is a valid JSON."

                vlm_response_text = self._get_vlm_response_multiview(
                    pil_labeled_rgb_image, view_images[0] if len(view_images) > 0 else None,
                    view_images[1] if len(view_images) > 1 else None,
                    view_images[2] if len(view_images) > 2 else None,
                    pil_map_image, prompt, self.img_buffer
                )

                parsed_json = robust_json_parse(vlm_response_text)

                if parsed_json and isinstance(parsed_json, dict) and 'action' in parsed_json:

                    if 'movement' in parsed_json:
                        self.instruction_obj.add_record_to_current_subtask({'movement': parsed_json['movement']})
                    if 'plan' in parsed_json:
                        self.instruction_obj.update_plan_for_current_subtask(parsed_json['plan'])
                    if 'completed' in parsed_json:
                        if parsed_json['completed'] is True:
                            return {'type': 'stop'}, vlm_response_text, pil_labeled_rgb_image

                    action_key = str(parsed_json['action']).upper()

                    if action_key == '-1':
                        return {'type': 'stop'}, vlm_response_text, pil_labeled_rgb_image

                    if action_key in actions:
                        return actions[action_key], vlm_response_text, pil_labeled_rgb_image
                    else:

                        continue
                else:

                    continue

            except Exception as e:
                pass

        return None, "None", pil_labeled_rgb_image

    def plan_rollback_path(self, subtask_key: str = None):
        """
        Plan a rollback path to a previous subtask start point (default: current subtask). Returns the rotation.
        Returns the planned waypoint sequence and rotation.
        :param subtask_key:
        :return:
        """
        if subtask_key is None:
            subtask_key = self.instruction_obj.get_current_subtask_key()

        if not subtask_key:
            pass
            return None, None

        try:
            target_idx = self.instruction_obj.sub_instruction_keys.index(subtask_key.upper())
        except ValueError:
            pass
            return None, None

        if target_idx == 0:

            target_pos = self.init_pos
            target_rot = self.init_rot
        else:

            prev_subtask_key = self.instruction_obj.sub_instruction_keys[target_idx - 1]
            target_pos, target_rot = self.instruction_obj.get_subtask_pos_rot_by_key(prev_subtask_key)

        if target_pos is None or target_rot is None:
            pass
            return None, None

        path = self.mapper.plan_path_to_target(target_pos)

        if path is None or len(path) == 0:
            pass
            return None, None

        return path, target_rot

    def _get_agent_yaw(self):
        """
        Extract yaw angle (around z) from self.mapper.current_rotation (3x3 rotation matrix).
        """
        R = np.asarray(self.mapper.current_rotation)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _rot_to_yaw(self, rotation):
        """
        Convert mapper.rotation to yaw angle.
        :param rotation:
        :return:
        """
        R = np.asarray(rotation)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _normalize_angle(self, a):
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    def step(self, step, rollback_step_threshold=70):
        """
        Execute one step based on prior agent decision; returns obs, top-down map, VLM response, labeled PIL image.
        """
        action_to_execute = None
        pil_labeled_img = None

        if not hasattr(self, '_position_history'):
            self._position_history = []
        if not hasattr(self, '_stuck_threshold'):
            self._stuck_threshold = 0.2
        if not hasattr(self, '_stuck_check_window'):
            self._stuck_check_window = 8
        if not hasattr(self, 'action_sequence'):
            self.action_sequence = []
        if not hasattr(self, 'current_path'):
            self.current_path = []

        if not hasattr(self, '_heading_before_nav'):
            self._heading_before_nav = 0.0
        if not hasattr(self, '_need_restore_heading'):
            self._need_restore_heading = False
        if not hasattr(self, '_no_move_steps'):
            self._no_move_steps = 0
        if not hasattr(self, '_global_stuck_limit'):
            self._global_stuck_limit = 20
        if not hasattr(self, '_no_move_epsilon'):
            self._no_move_epsilon = 1e-1

        current_pos = np.array(self.mapper.current_position[:2])
        self._position_history.append(current_pos)
        agent_state = self.curr_obs['agent_state']

        if len(self._position_history) > self._stuck_check_window + 1:
            self._position_history.pop(0)

        if self.prev_agent_position is not None:
            prev2d = np.array(self.prev_agent_position[:2])
            if np.linalg.norm(agent_state.position[:2] - prev2d) < self._no_move_epsilon:
                self._no_move_steps += 1
            else:
                self._no_move_steps = 0
        else:

            self._no_move_steps = 0

        if self._no_move_steps >= self._global_stuck_limit and action_to_execute is None:
            pass
            self.last_vlm_response = "(Auto)Global-stuck detected: stopping."
            self.vlm_responses.append(self.last_vlm_response)

            self.current_path = []
            self.action_sequence = []
            self._need_restore_heading = False
            action_to_execute = PolarAction.stop
            self.second_stop = True

        is_stuck = False
        if len(self._position_history) >= self._stuck_check_window:
            position_changes = [
                np.linalg.norm(self._position_history[i] - self._position_history[i - 1])
                for i in range(1, len(self._position_history))
            ]
            if len(position_changes) >= self._stuck_check_window and all(
                    change < self._stuck_threshold for change in position_changes[-self._stuck_check_window:]
            ):
                is_stuck = True
                print(
                    f"detected agent stuck, last{self._stuck_check_window} steps position changes: {position_changes[-self._stuck_check_window:]}")

                if self._need_restore_heading:
                    self.current_path = []
                    self.action_sequence = []

                # self._need_restore_heading = False

        if self._need_restore_heading and not self.current_path and not self.action_sequence:
            curr_yaw = self._get_agent_yaw()
            delta = - self._normalize_angle(self._heading_before_nav - curr_yaw)
            restore_threshold = math.radians(30)
            if abs(delta) > restore_threshold:
                if not self.rolling_back:
                    action_to_execute = PolarAction(r=0, theta=math.copysign(abs(delta) - restore_threshold, delta),
                                                type='turn')
                else:
                    action_to_execute = PolarAction(r=0, theta=delta, type='turn')
                    self.instruction_obj.current_step_count = 0
                    self.rolling_back = False

            self._need_restore_heading = False

        if action_to_execute is None and ((not self.current_path and not self.action_sequence) or is_stuck):
            if is_stuck:
                pass
                self.current_path, self.action_sequence, self._position_history = [], [], [current_pos]

            if self.instruction_obj.current_step_count <= rollback_step_threshold:
                decision, vlm_response, pil_labeled_img = self.decide_waypoint(
                    is_stuck=is_stuck, is_first_step=step == 2
                )
            else:
                decision, vlm_response, pil_labeled_img = {"type": "rollback", "subtask_key": self.instruction_obj.get_current_subtask_key()}, "(Auto)Exceeded step limit, initiating rollback to subtask start.", None

            self.vlm_responses.append(vlm_response)
            self.last_vlm_response = vlm_response
            self.img_buffer = []

            if decision is None:
                self._stop()
                action_to_execute = PolarAction.stop
                self.error = True

            elif decision['type'] == 'stop':
                if self.instruction_obj.is_last_subtask():
                    action_to_execute = PolarAction.stop
                    if self.first_stop:
                        self.instruction_obj.mark_current_subtask_completed(self.mapper.current_position - [0, 0, 1.2], self.mapper.current_rotation)
                    self._stop()
                else:
                    action_to_execute = PolarAction.pause()
                    self.instruction_obj.mark_current_subtask_completed(self.mapper.current_position - [0, 0, 1.2], self.mapper.current_rotation)

            elif decision['type'] == 'turn':
                action_to_execute = PolarAction(r=0, theta=decision['angle'], type='turn')
                self.current_path = []
                self._need_restore_heading = False

            elif decision['type'] == 'look_around':
                pass
                turn_right_action = PolarAction(r=0, theta=-self.turn_angle_rad*2, type='turn')
                self.action_sequence = [turn_right_action] * 2
                action_to_execute = self.action_sequence.pop(0)
                self._need_restore_heading = False

            elif decision['type'] == 'waypoint':
                target_world = decision['target_point_world']
                path = self.mapper.plan_path_to_target(target_world)
                self.target_world_position = target_world

                if path and len(path) > 1:

                    self._heading_before_nav = self._get_agent_yaw()
                    self._need_restore_heading = True

                    self.current_path = path[1:]
                    action_to_execute = self._get_action_for_next_waypoint()
                else:
                    pass
                    self.current_path = []
                    self._need_restore_heading = False
                    action_to_execute = PolarAction(r=0, theta=np.pi, type='turn')
            elif decision['type'] == 'rollback':

                subtask_key = decision.get('subtask_key', None)
                path, target_rot = self.plan_rollback_path(subtask_key=subtask_key)
                if path and len(path) > 1:
                    self.current_path = path[1:]
                    self.target_world_position = path[-1]

                    self._heading_before_nav = self._rot_to_yaw(target_rot)
                    self._need_restore_heading = True
                    action_to_execute = self._get_action_for_next_waypoint()
                    self.instruction_obj.add_record_to_current_subtask({"rollback": f"System: Too many steps have been taken without progress. Maybe you got lost? Rolling back to the start of subtask '{subtask_key}'."})
                    self.instruction_obj.current_step_count = 0
                    self.rolling_back = True
                else:
                    pass
                    self.current_path = []
                    self._need_restore_heading = False
                    action_to_execute = PolarAction(r=0, theta=np.pi, type='turn')
                    self.instruction_obj.current_step_count = 0

        elif action_to_execute is None and self.action_sequence:
            action_to_execute = self.action_sequence.pop(0)

        elif action_to_execute is None:
            action_to_execute = self._get_action_for_next_waypoint()

        self.instruction_obj.current_step_count += 1

        if action_to_execute is None:
            action_to_execute = PolarAction.null
            self.current_path = []

        self.execute_action(action_to_execute)

        if self.prev_agent_position is not None:
            self.traveled_distance += np.linalg.norm(agent_state.position - self.prev_agent_position)

        self.prev_agent_position = agent_state.position

        top_down_map = create_top_down_map_centered(
            self.mapper,
            self.config['camera']['fov'],
            self.target_world_position,
            action_candidates=self.last_actions,
            subtasks=self.instruction_obj.get_subtasks_key_coord()
        )

        if pil_labeled_img is None:
            current_image = self.curr_obs['color_sensor'][:, :, :3].copy()
            if action_to_execute is not None:
                action_text = self._action_to_text(action_to_execute)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)
                padding = 10
                (text_width, text_height), baseline = cv2.getTextSize(
                    action_text, font, font_scale, font_thickness
                )
                overlay = current_image.copy()
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (text_width + 2 * padding, text_height + 2 * padding + baseline),
                    bg_color,
                    -1
                )
                cv2.addWeighted(overlay, 0.6, current_image, 0.4, 0, current_image)
                cv2.putText(
                    current_image,
                    action_text,
                    (padding, text_height + padding),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )

            pil_labeled_img = Image.fromarray(current_image, 'RGB')
            top_down_map_rgb = cv2.cvtColor(top_down_map, cv2.COLOR_BGR2RGB)
            pil_map_image = Image.fromarray(top_down_map_rgb)
            self.img_buffer.append(pil_labeled_img)
            # self.img_buffer.append(pil_map_image)

        return self.curr_obs, top_down_map, self.last_vlm_response, pil_labeled_img

    def _action_to_text(self, action: PolarAction) -> str:
        """
        Convert PolarAction to English description.
        :param action:
        :return: English description of the action.
        """
        if action.type == 'stop':
            return "STOP"
        elif action.type == 'pause':
            return "PAUSE"
        elif action.type == 'null':
            return ""

        if abs(action.r) < 0.01:  # essentially no forward movement, pure rotation
            angle_deg = np.rad2deg(action.theta)
            if angle_deg > 0:
                return f"Turn Left {abs(angle_deg):.1f} degrees"
            elif angle_deg < 0:
                return f"Turn Right {abs(angle_deg):.1f} degrees"
            else:
                return "NO ROTATION"

        text = f"Move Forward {action.r:.2f}m"
        if abs(action.theta) > 0.01:
            angle_deg = np.rad2deg(action.theta)
            if angle_deg > 0:
                text += f" + Turn Left {abs(angle_deg):.1f} degrees"
            else:
                text += f" + Turn Right {abs(angle_deg):.1f} degrees"

        return text

    def _get_action_for_next_waypoint(self) -> Optional[PolarAction]:
        """
        Compute and return an action based on current position and the next path point.
        If a waypoint is reached, remove it from the path.
        The function prioritizes heading alignment before forward motion.
        """
        if not hasattr(self, 'current_path') or not self.current_path:
            return None

        agent_pos = self.mapper.current_position
        agent_rot_matrix = self.mapper.current_rotation
        next_waypoint = self.current_path[0]

        vec_to_waypoint_2d = np.array(next_waypoint[:2]) - agent_pos[:2]
        dist_to_waypoint = np.linalg.norm(vec_to_waypoint_2d)

        arrival_distance = 0.3  # arrival distance threshold (meters)
        if dist_to_waypoint < arrival_distance:
            self.current_path.pop(0)

            if not self.current_path:
                return None

            return self._get_action_for_next_waypoint()

        forward_vec_3d = -agent_rot_matrix[:, 2]
        forward_vec_2d = forward_vec_3d[:2]

        agent_angle = np.arctan2(forward_vec_2d[1], forward_vec_2d[0])
        waypoint_angle = np.arctan2(vec_to_waypoint_2d[1], vec_to_waypoint_2d[0])

        angle_diff = waypoint_angle - agent_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        angle_threshold_rad = np.deg2rad(5)  # 5 deg tolerance

        if abs(angle_diff) > angle_threshold_rad:

            return PolarAction(r=0, theta=-angle_diff, type='turn')
        else:

            max_forward_step = 1  # max forward distance per step (meters)
            forward_dist = min(dist_to_waypoint, max_forward_step)
            return PolarAction(r=forward_dist, theta=0, type='move_forward')

    def execute_action(self, action: PolarAction):
        """
        Execute the given action in the simulator and update the map.

        :param action:
        """
        obs = self.sim_wrapper.step(action)
        instances = self.update_map(obs)
        self.curr_obs = obs
        self.instances = instances
        self.last_action = action

    def create_action(self, world_point_1, world_point_2):
        """
        Create a PolarAction from world_point_1 to world_point_2.
        :param world_point_1: starting world coordinate.
        :param world_point_2: target world coordinate.
        :return: a PolarAction object.
        """

        delta = np.array(world_point_2) - np.array(world_point_1)
        r = np.linalg.norm([delta[0], delta[2]])
        theta = np.arctan2(delta[0], -delta[2])
        return PolarAction(r, theta)

    @abstractmethod
    def _get_vlm_response(self, front_rgb_image: PIL.Image.Image, map_image: PIL.Image.Image, prompt: str, img_buffer: list) -> str:
        """
        Call VLM API to get response for image and prompt.
        Subclasses must implement to call a specific VLM (e.g., GPT-4V).

        :param front_rgb_image:
        :param map_image:
        :param prompt:
        :param img_buffer:
        :return: VLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_vlm_response.")

    @abstractmethod
    def _get_vlm_response_multiview(self, front_rgb_image: PIL.Image.Image, left_rgb_image: PIL.Image.Image, right_rgb_image: PIL.Image.Image, back_rgb_image: PIL.Image.Image, map_image: PIL.Image.Image, prompt: str, img_buffer: list) -> str:
        """
        Call VLM API with multi-view images and prompt.
        Subclasses must implement to call a specific VLM (e.g., GPT-4V).

        :param front_rgb_image:
        :param left_rgb_image:
        :param right_rgb_image:
        :param back_rgb_image:
        :param map_image:
        :param prompt:
        :param img_buffer:
        :return: VLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_vlm_response_multiview.")

    @abstractmethod
    def _get_llm_response(self, prompt: str) -> str:
        """
        Call text LLM API for prompt response.
        Subclasses must implement to call a specific LLM.

        :param prompt:
        :return: LLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_llm_response.")

    def generate_prompt(self, instruction: str = None, is_stuck: bool = False, is_first_step=False) -> str:
        """
        Generate an English prompt describing agent state, map, and available actions.
        This prompt is optimized for PathPlannerAgent decision logic.
        """
        pos = self.mapper.current_position
        pos_str = f"Your current coordinate (z is height): (x: {pos[0]:.2f}, y: {pos[1]:.2f}, z: {pos[2]:.2f})"

        after_stop_str = """**IMPORTANT: Blue arc indicates a 1-meter range around the agent. Check again if you are as close as possible to the target position(<1m). If not, continue to get closer. If you are close enough, return "action": -1 to stop.**"""

        prompt_objnav_1st = f"""
    You are an intelligent agent in a simulated indoor environment. Your high-level instruction is: {instruction}.
    {pos_str}
    {"Your position hasn't changed for a long time, possibly stuck or in a loop, so you need to reconsider your next move carefully." if is_stuck else ""}
    Your task is to choose a strategic waypoint or a turn action. Once you choose a waypoint, a low-level planner will automatically generate a path and navigate to it.

    **Image Inputs**:
    1.  **Top-Down Map**: This is your memory. It will be updated as you explore. It's oriented with you facing upwards.
        -   `Gray`: Navigable floor you have seen.
        -   `Black`: Obstacles or walls that cannot be passed.
        -   `Red Arrow`: Your current position and direction.
        -   `Red Trail`: Your recent trajectory.
        -   `Start Point`: Your starting position.
        -   `Text Labels`: Automatically detected objects (may be inaccurate).
        -   `Yellow Boundary`: The frontier between explored and unexplored areas.
        -   `Numbered Circles`: Candidate waypoints projected into your view.
    2.  **First-Person View**: This is what you see right now. It is divided into 3 perspectives: front, left, right.
        -   `Numbered Circles`: Candidate waypoints projected into your front view.
        -   `L, R, B Circles`: Turn actions (L: Left 90°, R: Right 90°, B: Turn Around 180°).
        -   `Action Text`: The last action you executed.   

    **Notes**:
    1. Labels on the map are sometimes wrong, so stick to the first-person view.
    2. You can't walk through or open a closed door. And the target will not be in these places.
    3. When you don't know where to go, prioritize exploring unexplored areas.
    4. Don't go up or down stairs.
    5. Red waypoints indicate locations that have been visited, and white ones indicate those that haven’t.

    **Your Task**:
    1.  **Analyze**: Briefly describe your current situation and environment, referencing both the map and your first-person view. Confirm your current progress on the instruction. Determine whether a previous decision or judgment was correct. 
    2.  **Strategize**: State your plan to make progress on the instruction. Which direction or area should you explore next? Pay attention to unexplored areas. When you're not sure where to go, you can turn your perspective and look around.
    3.  **Decide**: Choose the best action to execute your plan. Prioritize waypoints that lead towards the goal or into new, unexplored areas. Avoid choosing waypoints that require navigation through tight spaces or closed doors.

    **Output Format**:
    ```json
    {{
    "movement": "Describe the movement trajectory of the last step based on your last action and historical frames",
    "observation": {{   // According to the latest view and map, describe the scene you see and your position currently.
        "front view": "",
        "left view": "",
        "right view": "",
        "map": "",
    }},
    "thought": "",  // Check whether the current route and position are consistent with the instruction and previous plan. Judge if you are close to the target or need further exploration
    "stuck": true/false,  // Check if you're blocked by obstacles or going in circles.
    "plan": "",  // For subtask {self.instruction_obj.get_current_subtask_key()}, imagine the approximate final position when the subtask is completed (near the target object). Give a short high-level route plan WITHOUT mentioning specific waypoint numbers.
    "curr_step": "",  // Check if the subtask {self.instruction_obj.get_current_subtask_key()} is completed(reach the closest waypoint to the target). Analyze exact candidate waypoints with number and turning actions. Make a decision for the current step.
    "action": ""  // Return action -1 to complete if you reach the expected destination of the subtask {self.instruction_obj.get_current_subtask_key()}. Otherwise, select a waypoint(if available) on the image or turning action(L, R, B). For example: "action": 3 or "action": "L".
    }}
    ```
    """.strip()

        prompt_objnav = f"""
        **Continue your task.**
        Your high-level instruction is: {instruction}.
        {pos_str}
        {"Your position hasn't changed for a long time, possibly stuck or in a loop, so you need to reconsider your next move carefully." if is_stuck else ""}
        Your task is to choose a strategic waypoint or a turn action. Once you choose a waypoint, a low-level planner will automatically generate a path and navigate to it.

        **Your Task**:
        1.  **Analyze**: Briefly describe your current situation and environment, referencing both the map and your first-person view. Confirm your current progress on the instruction. Determine whether a previous decision or judgment was correct. 
        2.  **Strategize**: State your plan to make progress on the instruction. Which direction or area should you explore next? Pay attention to unexplored areas. When you're not sure where to go, you can turn your perspective and look around.
        3.  **Decide**: Choose the best action to execute your plan. Prioritize waypoints that lead towards the goal or into new, unexplored areas. Avoid choosing waypoints that require navigation through tight spaces or closed doors.
        {after_stop_str if self.first_stop else ""}
        **Output Format**:
        ```json
        {{
        "movement": "Describe the movement trajectory of the last step based on your last action and historical frames",
        "observation": {{   // According to the latest view and map, describe the scene you see and your position currently.
            "front view": "",
            "left view": "",
            "right view": "",
            "map": "",
        }},
        "thought": "",  // Check whether the current route and position are consistent with the instruction and previous plan. Judge if you are close to the target or need further exploration
        "stuck": true/false,  // Check if you're blocked by obstacles or going in circles.
        "plan": "",  // For subtask {self.instruction_obj.get_current_subtask_key()}, imagine the approximate final position when the subtask is completed (near the target object). Give a short high-level route plan WITHOUT mentioning specific waypoint numbers.
        "curr_step": "",  // Check if the subtask {self.instruction_obj.get_current_subtask_key()} is completed(reach the closest waypoint to the target). Analyze exact candidate waypoints with number and turning actions. Make a decision for the current step.
        "action": ""  // Return action -1 to complete if you reach the expected destination of the subtask {self.instruction_obj.get_current_subtask_key()}. Otherwise, select a waypoint(if available) on the image or turning action(L, R, B). For example: "action": 3 or "action": "L".
        }}
        ```""".strip()

        prompt_vln_1st = f"""
        **Your Task**:
        {self.instruction_obj.get_all_subtasks_str()}

        ---
        The instruction describes a path from the starting position to the target position. Your task is to move from the starting position(0,0,0) to the final position. At the same time, make sure your route conforms to the instruction description.
        The "Go upstairs" in the instruction is only considered complete when you completely reach the top platform via the stairs. Make sure you have climbed all the steps and moved onto the platform.

        {pos_str}
        Before starting to execute the instruction, you have firstly turned around 2 times in place for a full 360 degrees to capture images of your surroundings.
        Your task is to choose a strategic waypoint or a turn action. Once you choose a waypoint, a low-level planner will automatically generate a path and navigate to it.
        Complete subtasks in order. Don't skip any action. Consider the route by taking into account the current and the next subtask.

        **Image Inputs**:
        1.  **Top-Down Map**: This is your memory. It will be updated as you explore. It's oriented with you facing upwards.
            -   `Gray`: Navigable floor you have seen.
            -   `Black`: Obstacles or walls that cannot be passed.
            -   `Red Arrow`: Your current position and direction.
            -   `Red Trail`: Your recent trajectory.
            -   `Start Point`: Your starting position.
            -   `Text Labels`: Automatically detected objects (may be inaccurate).
            -   `Yellow Boundary`: The frontier between explored and unexplored areas.
            -   `Numbered Circles`: Candidate waypoints projected into your view.
        2.  **First-Person View**: This is what you see right now. It is divided into 3 perspectives: front, left, right.
            -   `Numbered Circles`: Candidate waypoints projected into your front view.
            -   `L, R, B Circles`: Turn actions (L: Left 90°, R: Right 90°, B: Turn Around 180°).
            -   `Action Text`: The last action you executed.          

        **Important Notes**:
        1. Labels on the map can be wrong; trust your first-person view more.
        2. You cannot walk through or open closed doors. The target will not be in such places. There's no need to try to open doors.
        3. You can turn right/left/round in place to observe the surrounding environment.
        4. Avoid going back to places you've already been unless necessary!!!
        5. There's no need to operate objects, just move to the target position.
        6. Red waypoints indicate locations that have been visited, and white ones indicate those that haven’t.

        **Output Format**:
        ```json
        {{
        "observation": {{   // According to the latest view and map, describe the scene you see and your position currently.
            "front view": "",
            "left view": "",
            "right view": "",
            "map": "",
        }},
        "thought": "",  // Check whether the current route and position are in line with the instruction and previous plan and whether the expected destination of the subtask has been reached, and plan the next action(turn, move forward or complete).
        "stuck": true/false,  // Check if you're blocked by obstacles or going in circles.
        "plan": "",  // For subtask {self.instruction_obj.get_current_subtask_key()}, based on the screen and instructions, envision the target position you expect to reach when the subtask is completed, and plan the route. Don't mention exact waypoint number here.
        "curr_step": "",  // Check if the subtask {self.instruction_obj.get_current_subtask_key()} is completed(reach the closest waypoint to the target). Analyze exact candidate waypoints with number and turning actions. Make a decision for the current step.
        "action": ""  // Return action -1 to complete if you reach the expected destination of the subtask {self.instruction_obj.get_current_subtask_key()}. Otherwise, select a waypoint(if available) on the image or turning action(L, R, B). For example: "action": 3 or "action": "L".
        }}
        ```""".strip()
        prompt_vln = f"""
        **Continue your task.**
        {self.instruction_obj.get_all_subtasks_str()}

        {"You can't move as expected, possibly stuck or blocked by something, try to find other ways to get out." if is_stuck else ""}

        ---
        The "Go upstairs" in the instruction is only considered complete when you completely reach the top platform via the stairs. Make sure you have climbed all the steps and moved onto the platform.

        {pos_str}
        {"Your position hasn't changed after multiple retries, possibly blocked by something, find out the reason why you are stuck and try to turn and look for other ways to get out." if is_stuck else ""}
        Your task is to choose a strategic waypoint or a turn action. Once you choose a waypoint, a low-level planner will automatically generate a path and navigate to it.
        Red waypoints indicate locations you have visited. Avoid going back to places you've already been unless necessary!!!
        Complete subtasks in order. Don't skip any action. Consider the route by taking into account the current and the next subtask.
        {after_stop_str if self.first_stop else ""}
        **Output Format**:
        ```json
        {{
        "movement": "Describe the movement trajectory of the last step based on your last action and historical frames",
        "observation": {{   // According to the latest view and map, describe the scene you see and your position currently.
            "front view": "",
            "left view": "",
            "right view": "",
            "map": "",
        }},
        "thought": "",  // Check whether the current route and position are in line with the instruction and previous plan and whether the expected destination of the subtask has been reached, and plan the next action(turn, move forward or complete).
        "stuck": true/false,  // Check if you're blocked by obstacles or going in circles.
        "plan": "",  // For subtask {self.instruction_obj.get_current_subtask_key()}, based on the screen and instructions, envision the target position you expect to reach when the subtask is completed, and plan the route. Don't mention exact waypoint number here.
        "curr_step": "",  // Check if the subtask {self.instruction_obj.get_current_subtask_key()} is completed(reach the closest waypoint to the target). Analyze exact candidate waypoints with number and turning actions. Make a decision for the current step.
        "action": ""  // Return action -1 to complete if you reach the expected destination of the subtask {self.instruction_obj.get_current_subtask_key()}. Otherwise, select a waypoint(if available) on the image or turning action(L, R, B). For example: "action": 3 or "action": "L".
        }}
        ```""".strip()

        if self.mode == 'objnav':
            prompt = prompt_objnav if not is_first_step else prompt_objnav_1st
        else:
            prompt = prompt_vln if not is_first_step else prompt_vln_1st
        return prompt

    def decompose_instruction(self, instruction: str) -> List[str]:
        """
        Call LLM to decompose a complex instruction into subgoals. Parses JSON {"subgoals": [...]} first;
        falls back to heuristic splitting on failure. Returns list of subgoal strings.
        """
        prompt = """
        Here is a visual language navigation instruction. The agent only needs to move to the designated position according to the instruction and does not operate any object.
        Decompose the following navigation instruction. Break down a complex instruction into multiple subtasks in sequence.
        The decomposition principle requires ensuring each subtask has a clear completion condition(end position); otherwise, it cannot be an independent subtask and you can combine multiple subtasks into one subtask if necessary.
        Each subtask must contain at least 3 objects or places to ensure clarity. Under the above condition, each sub-task should be as small as possible.
        **You can only split and merge based on the original instructions, and must not alter the original expression.** Do not add any additional explanations.
        For example, only "walk through the hallway" lacks a clear end position and should therefore be merged with subsequent subtasks until it's clear. But a subtask cannot contain more than three actions.
        First, analyze the instruction and write a draft.
        Finally, output in JSON format. Return JSON in the exact form: {\"analysis\": \"\", \"subtasks\": [\"...\", \"...\", ...]}\n\n
        SELF-CHECK BEFORE OUTPUT:
        - For every subtask, verify it is an exact substring of the original instruction.
        - Count objects/places (must be >= 2).
        - Count actions (must be <= 3).
        - If a subtask violates rules, MERGE it with adjacent text until it satisfies all rules.

        Instruction: {instruction}\n\n
        """

        prompt = prompt.replace("{instruction}", instruction)

        try:
            raw = self._get_llm_response(prompt)
        except Exception as e:
            raw = ""
        subs: List[str] = []

        try:
            m = re.search(r'\{.*\}', raw, flags=re.S)
            candidate = m.group(0) if m else raw
            data = json.loads(candidate)
            if isinstance(data, dict) and "subtasks" in data and isinstance(data["subtasks"], list):
                for s in data["subtasks"]:
                    if not isinstance(s, str):
                        continue

                    s_clean = re.sub(r'^\s*\d+\s*[\)\.\-:]*\s*', '', s).strip()
                    if s_clean:
                        subs.append(s_clean)
        except Exception:
            subs = []

        if not subs:
            parts = re.split(r'(?:then|and then|after that|;|\n|,|\.)+', instruction, flags=re.I)
            candidates = [p.strip() for p in parts if p and p.strip()]

            merged: List[str] = []
            for p in candidates:
                if len(p) < 6 and merged:
                    merged[-1] = (merged[-1] + ' ' + p).strip()
                else:
                    merged.append(p)
            subs = merged if merged else [instruction.strip()]

        if not subs:
            subs = [instruction.strip()]

        print(subs)

        return subs

    def close(self):
        """Close and release GPU memory/large object refs; safe to call repeatedly."""
        if getattr(self, "_ended", False):
            return

        try:
            if hasattr(self, "mapper") and self.mapper is not None:

                try:
                    release = getattr(self.mapper, "release_all_vram", None)
                    if callable(release):
                        release()
                except Exception:
                    pass
                finally:

                    self.mapper = None
        except Exception:
            pass

        for attr in ("curr_obs", "instances", "last_actions", "img_buffer"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        self._ended = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

import base64
from io import BytesIO
from openai import AzureOpenAI, OpenAI

class GPTAgent(PathPlannerAgent):
    """
    VLM agent implementation using OpenAI / Azure OpenAI compatible API.
    Configured via environment variables:
      - OPENAI_API_KEY   API key
      - OPENAI_BASE_URL  (optional) custom base_url
      - AZURE_OPENAI_ENDPOINT   Azure endpoint
      - AZURE_OPENAI_API_KEY   Azure API key
      - AZURE_OPENAI_API_VERSION  Azure API version
    """

    def __init__(self, sim_wrapper, config, instruction, initial_position=None, initial_rotation=None, model_name=None, mode='vln'):
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-5")

        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        if azure_endpoint and azure_api_key:
            self.client = AzureOpenAI(api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=azure_api_key)
        else:
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        sys_prompt = """

        """
        self.history_msgs = [
            {"role": "system", "content": "You are an agent good at navigating in an indoor environment."}
        ]
        super().__init__(sim_wrapper, config, instruction, initial_position, initial_rotation, mode)

    def _pil_to_base64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def _get_vlm_response(self, rgb_image: Image.Image, map_image: Image.Image, prompt: str, img_buffer=None) -> str:
        if img_buffer is None:
            img_buffer = []
        try:
            rgb_img_b64 = self._pil_to_base64(rgb_image)
            map_img_b64 = self._pil_to_base64(map_image)

            content_for_request = [{"type": "text", "text": prompt}]
            for img in img_buffer:
                img_b64 = self._pil_to_base64(img)
                content_for_request.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
                })
            content_for_request.extend([
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_img_b64}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "high"}}
            ])
            current_msg_for_api = {"role": "user", "content": content_for_request}

            content_for_history = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_img_b64}", "detail": "auto"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "auto"}},
            ]
            current_msg_for_history = {"role": "user", "content": content_for_history}

            if not hasattr(self, "history_msgs"):
                self.history_msgs = [
                    {"role": "system", "content": "You are an agent good at navigating in an indoor environment."}
                ]

            messages_for_api = self.history_msgs + [current_msg_for_api]

            user_msgs = [msg for msg in messages_for_api if msg["role"] == "user"]
            if len(user_msgs) > 7:
                user_count = 0
                for msg in messages_for_api:
                    if msg["role"] == "user":
                        user_count += 1
                        if user_count <= len(user_msgs) - 7:
                            if isinstance(msg["content"], list):
                                msg["content"] = [c for c in msg["content"] if c["type"] == "text"]

            max_turn = 30
            if len(messages_for_api) > max_turn:
                messages_for_api = messages_for_api[-max_turn:]

                if messages_for_api[0]["role"] != "system":
                    messages_for_api.insert(0, {"role": "system",
                                                "content": "You are an agent good at navigating in an indoor environment."})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,

            )

            result = response.choices[0].message.content
            print(result)

            self.history_msgs.append(current_msg_for_history)
            self.history_msgs.append({
                "role": "assistant",
                "content": result
            })
            return result
        except Exception as e:
            pass
            return "stop"

    def _get_vlm_response_multiview(self, front_rgb_image: PIL.Image.Image, left_rgb_image: PIL.Image.Image,
                                    right_rgb_image: PIL.Image.Image, back_rgb_image: PIL.Image.Image,
                                    map_image: PIL.Image.Image, prompt: str, img_buffer: list) -> str:
        if img_buffer is None:
            img_buffer = []
        try:

            front_img_b64 = self._pil_to_base64(front_rgb_image)
            left_img_b64 = self._pil_to_base64(left_rgb_image)
            right_img_b64 = self._pil_to_base64(right_rgb_image)
            # back_img_b64 = self._pil_to_base64(back_rgb_image)
            map_img_b64 = self._pil_to_base64(map_image)

            content_for_request = [{"type": "text", "text": prompt}]
            for img in img_buffer[-20:]:
                img_b64 = self._pil_to_base64(img)
                content_for_request.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
                })

            content_for_request.extend([

                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{left_img_b64}", "detail": "high"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{right_img_b64}", "detail": "high"}},
                # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{back_img_b64}", "detail": "high"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{front_img_b64}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "high"}},
            ])
            current_msg_for_api = {"role": "user", "content": content_for_request}

            content_for_history = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{front_img_b64}", "detail": "auto"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "auto"}},
            ]
            current_msg_for_history = {"role": "user", "content": content_for_history}

            sys_prompt = """
                    **Image Inputs**:
        1.  **Top-Down Map**: This is your memory. It will be updated as you explore. It's oriented with you facing upwards.
            -   `Gray`: Navigable floor you have seen.
            -   `Black`: Obstacles or walls that cannot be passed.
            -   `Red Arrow`: Your current position and direction.
            -   `Red Trail`: Your recent trajectory.
            -   `Start Point`: Your starting position.
            -   `Text Labels`: Automatically detected objects (may be inaccurate).
            -   `Yellow Boundary`: The frontier between explored and unexplored areas.
            -   `Numbered Circles`: Candidate waypoints projected into your view.
        2.  **First-Person View**: This is what you see right now. It is divided into 3 perspectives: front, left, right.
            -   `Numbered Circles`: Candidate waypoints projected into your front view.
            -   `L, R, B Circles`: Turn actions (L: Left 90°, R: Right 90°, B: Turn Around 180°).
            -   `Action Text`: The last action you executed.

        **Important Notes**:
        1. Labels on the map can be wrong; trust your first-person view more.
        2. You cannot walk through or open closed doors. The target will not be in such places. There's no need to try to open doors.
        3. You can turn right/left/round in place to observe the surrounding environment.
        4. Avoid going back to places you've already been unless necessary!!!
        5. There's no need to operate objects, just move to the target position.
        6. Red waypoints indicate locations that have been visited, and white ones indicate those that haven’t.
            """

            if not hasattr(self, "history_msgs"):
                self.history_msgs = [
                    {"role": "system", "content": sys_prompt}
                ]

            messages_for_api = self.history_msgs + [current_msg_for_api]

            user_msgs = [msg for msg in messages_for_api if msg["role"] == "user"]
            if len(user_msgs) > 8:
                user_count = 0
                for msg in messages_for_api:
                    if msg["role"] == "user":
                        user_count += 1
                        if user_count <= len(user_msgs) - 8:
                            msg["content"] = [c for c in msg["content"] if c["type"] == "text"]

            max_turn = 30
            if len(messages_for_api) > max_turn:
                messages_for_api = messages_for_api[-max_turn:]
                if messages_for_api[0]["role"] != "system":
                    messages_for_api.insert(0, {"role": "system",
                                                "content": "You are an agent good at navigating in an indoor environment."})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                # temperature=0.9,
                # extra_body={"vl_high_resolution_images": False},
            )
            result = response.choices[0].message.content
            print(result)

            self.history_msgs.append(current_msg_for_history)
            self.history_msgs.append({
                "role": "assistant",
                "content": result
            })
            return result
        except Exception as e:
            pass
            return "stop"

    def _get_llm_response(self, prompt: str) -> str:
        """
        Call the text LLM (no image, no chat history) for instruction decomposition.
        Returns the model's plain-text response, or "stop" on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(response, "choices") and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content
                if hasattr(choice, "text"):
                    return choice.text
                return str(choice)
            if hasattr(response, "output_text"):
                return response.output_text
            return str(response)
        except Exception as e:
            pass
            return "stop"
