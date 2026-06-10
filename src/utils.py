import base64
import io
import logging
import os
import traceback
import cv2
import numpy as np
import math
from PIL import Image
import quaternion

import matplotlib.pyplot as plt
import matplotlib.animation as animation

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (200, 200, 200)

def local_to_global(position, orientation, local_point):
    """
    Transforms a local coordinate point to global coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        local_point (np.ndarray): The point in local coordinates.

    Returns:
        np.ndarray: Transformed global coordinates.
    """
    rotated_point = quaternion.rotate_vectors(orientation, local_point)
    global_point = rotated_point + position
    return global_point

def global_to_local(position, orientation, global_point):
    """
    Transforms a global coordinate point to local coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        global_point (np.ndarray): The point in global coordinates.

    Returns:
        np.ndarray: Transformed local coordinates.
    """
    translated_point = global_point - position
    inverse_orientation = np.quaternion.conj(orientation)
    local_point = quaternion.rotate_vectors(inverse_orientation, translated_point)
    return local_point

def calculate_focal_length(fov_degrees, image_width):
    """
    Calculates the focal length in pixels based on the field of view and image width.

    Args:
        fov_degrees (float): Field of view in degrees.
        image_width (int): The width of the image in pixels.

    Returns:
        float: The focal length in pixels.
    """
    fov_radians = np.deg2rad(fov_degrees)
    focal_length = (image_width / 2) / np.tan(fov_radians / 2)
    return focal_length

def local_to_image(local_point, resolution, focal_length):
    """
    Converts a local 3D point to image pixel coordinates.

    Args:
        local_point (np.ndarray): The point in local coordinates.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The pixel coordinates (x_pixel, y_pixel).
    """
    point_3d = [local_point[0], -local_point[1], -local_point[2]]  # Inconsistency between Habitat camera frame and classical convention
    if point_3d[2] == 0:
        point_3d[2] = 0.0001
    x = focal_length * point_3d[0] / point_3d[2]
    x_pixel = int(resolution[1] / 2 + x)

    y = focal_length * point_3d[1] / point_3d[2]
    y_pixel = int(resolution[0] / 2 + y)
    return x_pixel, y_pixel

def unproject_2d(x_pixel, y_pixel, depth, resolution, focal_length):
    """
    Unprojects a 2D pixel coordinate back to 3D space given depth information.

    Args:
        x_pixel (int): The x coordinate of the pixel.
        y_pixel (int): The y coordinate of the pixel.
        depth (float): The depth value at the pixel.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The 3D coordinates (x, y, z).
    """
    x = (x_pixel - resolution[1] / 2) * depth / focal_length
    y = (y_pixel - resolution[0] / 2) * depth / focal_length
    return x, -y, -depth

def agent_frame_to_image_coords(point, agent_state, sensor_state, resolution, focal_length):
    """
    Converts a point from agent frame to image coordinates.

    Args:
        point (np.ndarray): The point in agent frame coordinates.
        agent_state (6dof): The agent's state containing position and rotation.
        sensor_state (6dof): The sensor's state containing position and rotation.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple or None: The image coordinates (x_pixel, y_pixel), or None if the point is behind the camera.
    """
    global_p = local_to_global(agent_state.position, agent_state.rotation, point)
    camera_pt = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
    if camera_pt[2] > 0:
        return None
    return local_to_image(camera_pt, resolution, focal_length)

def put_text_on_image(image, text, location, custom_pos=None, font=cv2.FONT_HERSHEY_SIMPLEX, text_size=2.7, bg_color=(255, 255, 255),
                      text_color=(0, 0, 0), text_thickness=3, highlight=True):
    """
    Puts text on an image with optional background highlighting.

    Args:
        image (np.ndarray): The image to draw on.
        text (str): The text to put on the image.
        location (str): Position for the text ('top_left', 'top_right', 'bottom_left', etc., or 'custom').
        custom_pos (tuple): The (x, y) coordinates for the text if location is 'custom'.
        font (int): Font to use for the text.
        text_size (float): Size of the text.
        bg_color (tuple): Background color for the text (BGR).
        text_color (tuple): Color of the text (BGR).
        text_thickness (int): Thickness of the text font.
        highlight (bool): Whether to highlight the text background.

    Returns:
        np.ndarray: Image with text added.
    """
    scale_factor = image.shape[0] / 1080
    adjusted_thickness = math.ceil(scale_factor * text_thickness)
    adjusted_size = scale_factor * text_size

    valid_locations = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_center', 'center', 'custom']
    assert location in valid_locations, f"Invalid location. Choose from {valid_locations}"
    if location == 'custom':
        assert custom_pos is not None, "custom_pos must be provided when location is 'custom'"

    img_height, img_width = image.shape[:2]
    text_size_cv, _ = cv2.getTextSize(text, font, adjusted_size, adjusted_thickness)

    # Calculate text position
    offset = math.ceil(10 * scale_factor)
    text_x, text_y = 0, 0

    if location == 'custom':
        text_x, text_y = custom_pos
    elif location == 'top_left':
        text_x, text_y = offset, text_size_cv[1] + offset
    elif location == 'top_right':
        text_x, text_y = img_width - text_size_cv[0] - offset, text_size_cv[1] + offset
    elif location == 'bottom_left':
        text_x, text_y = offset, img_height - offset
    elif location == 'bottom_right':
        text_x, text_y = img_width - text_size_cv[0] - offset, img_height - offset
    elif location == 'top_center':
        text_x, text_y = (img_width - text_size_cv[0]) // 2, text_size_cv[1] + offset
    elif location == 'center':
        text_x, text_y = (img_width - text_size_cv[0]) // 2, (img_height + text_size_cv[1]) // 2

    # Draw background rectangle
    if highlight:
        # Adjust background rectangle to center the text
        top_left = (text_x - offset // 2, text_y - text_size_cv[1] - offset)
        bottom_right = (text_x + text_size_cv[0] + offset // 2, text_y + offset)
        cv2.rectangle(image, top_left, bottom_right, bg_color, -1)

    # Add the text
    cv2.putText(image, text, (text_x, text_y), font, adjusted_size, text_color, adjusted_thickness)
    return image

def find_intersections(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int):
    """
    Find the intersections of a line defined by two points with the image boundaries.
    Args:
        x1 (int): The x-coordinate of the first point.
        y1 (int): The y-coordinate of the first point.
        x2 (int): The x-coordinate of the second point.
        y2 (int): The y-coordinate of the second point.
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        list of tuple or None: A list of two tuples representing the intersection points 
        with the image boundaries, or None if there are not exactly two intersections.
    """
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m = None  # Vertical line
        b = None

    intersections = []
    if m is not None and m != 0:  # Avoid division by zero for horizontal lines
        x_at_yh = int((img_height - b) / m)  # When y = img_height, x = (img_height - b) / m
        if 0 <= x_at_yh <= img_width:
            intersections.append((x_at_yh, img_height - 1))

    if m is not None:
        y_at_x0 = int(b)  # When x = 0, y = b
        if 0 <= y_at_x0 <= img_height:
            intersections.append((0, y_at_x0))

    if m is not None:
        y_at_xw = int(m * img_width + b)  # When x = img_width, y = m * img_width + b
        if 0 <= y_at_xw <= img_height:
            intersections.append((img_width - 1, y_at_xw))

    if m is not None and m != 0:  # Avoid division by zero for horizontal lines
        x_at_y0 = int(-b / m)  # When y = 0, x = -b / m
        if 0 <= x_at_y0 <= img_width:
            intersections.append((x_at_y0, 0))

    if m is None:
        intersections.append((x1, img_height - 1))  # Bottom edge
        intersections.append((x1, 0))  # Top edge

    if len(intersections) == 2:
        return intersections
    return None

def depth_to_height(depth_image, hfov, camera_position, camera_orientation):
    """
    Converts depth image to a height map using camera parameters.

    Args:
        depth_image (np.ndarray): The input depth image.
        hfov (float): Horizontal field of view in degrees.
        camera_position (np.ndarray): The global position of the camera.
        camera_orientation (quaternion.quaternion): The camera's quaternion orientation.

    Returns:
        np.ndarray: Global height map derived from depth image.
    """
    img_height, img_width = depth_image.shape
    focal_length_px = img_width / (2 * np.tan(np.radians(hfov / 2)))

    i_idx, j_idx = np.indices((img_height, img_width))
    x_prime = (j_idx - img_width / 2)
    y_prime = (i_idx - img_height / 2)

    x_local = x_prime * depth_image / focal_length_px
    y_local = y_prime * depth_image / focal_length_px
    z_local = depth_image

    local_points = np.stack((x_local, -y_local, -z_local), axis=-1)
    global_points = local_to_global(camera_position, camera_orientation, local_points)

    return global_points[:, :, 1]  # Return height map

def log_exception(e):
    """Logs an exception with traceback information."""
    tb = traceback.extract_tb(e.__traceback__)
    for frame in tb:
        logging.error(f"Exception in {frame.filename} at line {frame.lineno}")
    logging.error(f"Error: {e}")

def create_gif(image_dir, interval=600):
    """
    Creates a GIF animation from images in the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.
        interval (int): Interval between frames in milliseconds.

    Returns:
        None: Saves the GIF animation in the directory.
    """
    # Create a figure that tightly matches the size of the images (1920x1080)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.set_position([0, 0, 1, 1])  # Remove all padding
    ax.axis('off')

    frames = []

    # Process up to 80 steps
    for i in range(min(len(os.listdir(image_dir)) - 1, 80)):
        try:
            img = cv2.imread(f"{image_dir}/step{i}/color_sensor.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)

            img_copy = cv2.imread(f"{image_dir}/step{i}/color_sensor_chosen.png")
            img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            frame_copy = [ax.imshow(img_copy_rgb, animated=True)]
            frames.append(frame_copy)

        except Exception as e:
            continue

    # Add a black frame at the end
    black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
    frame_black = [ax.imshow(black_frame_rgb, animated=True)]
    frames.append(frame_black)

    # Create the animation
    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)

    # Save the animation
    ani.save(f'{image_dir}/animation.gif', writer='imagemagick')
    logging.info('GIF animation saved successfully!')

def encode_image_b64(image: Image.Image, format="PNG") -> str:
    """
    Convert a PIL Image to Base64 string.

    Parameters:
        image (Image.Image): The PIL image to convert.

    Returns:
        image_b64: The image in base64 as a string.
    """
    with io.BytesIO() as output:
        image.save(output, format=format)
        return base64.b64encode(output.getvalue()).decode(('utf-8'))

def append_mime_tag(image_b64, format="png"):
    return f"data:image/{format};base64,{image_b64}"

def resize_image_if_needed(image: Image.Image, max_dimension: int) -> Image.Image:
    """
    Resize a PIL image to ensure that its largest dimension does not exceed max_size.

    Parameters:
        image (Image.Image): The PIL image to resize.
        max_size (int): The maximum size for the largest dimension.

    Returns:
        Image.Image: The resized image.
    """
    width, height = image.size

    # Check if the image has a palette and convert it to true color mode
    if image.mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")

    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)

    return image

import json5
import re
import json

def _strip_markdown_code_fence(text: str) -> str:
    """Remove a single outer Markdown code fence when the whole text is fenced."""
    stripped = text.strip()
    match = re.fullmatch(r'```(?:json|json5)?\s*(.*?)\s*```', stripped, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else stripped

def _markdown_json_blocks(text: str):
    for match in re.finditer(r'```(?:json|json5)?\s*(.*?)\s*```', text, flags=re.IGNORECASE | re.DOTALL):
        candidate = match.group(1).strip()
        if candidate:
            yield candidate

def _balanced_json_spans(text: str):
    """Yield balanced JSON-looking object/array spans embedded in free text."""
    pairs = {'{': '}', '[': ']'}
    closers = set(pairs.values())
    quote = None
    escape = False
    line_comment = False
    block_comment = False

    for start, ch in enumerate(text):
        if line_comment:
            if ch in '\r\n':
                line_comment = False
            continue
        if block_comment:
            if ch == '*' and start + 1 < len(text) and text[start + 1] == '/':
                block_comment = False
            continue
        if quote:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            continue
        if ch == '/' and start + 1 < len(text):
            nxt = text[start + 1]
            if nxt == '/':
                line_comment = True
                continue
            if nxt == '*':
                block_comment = True
                continue
        if ch not in pairs:
            continue

        stack = [pairs[ch]]
        quote2 = None
        escape2 = False
        line_comment2 = False
        block_comment2 = False
        idx = start + 1

        while idx < len(text):
            curr = text[idx]
            nxt = text[idx + 1] if idx + 1 < len(text) else ''

            if line_comment2:
                if curr in '\r\n':
                    line_comment2 = False
                idx += 1
                continue
            if block_comment2:
                if curr == '*' and nxt == '/':
                    block_comment2 = False
                    idx += 2
                else:
                    idx += 1
                continue
            if quote2:
                if escape2:
                    escape2 = False
                elif curr == '\\':
                    escape2 = True
                elif curr == quote2:
                    quote2 = None
                idx += 1
                continue
            if curr in ('"', "'"):
                quote2 = curr
                idx += 1
                continue
            if curr == '/' and nxt == '/':
                line_comment2 = True
                idx += 2
                continue
            if curr == '/' and nxt == '*':
                block_comment2 = True
                idx += 2
                continue
            if curr in pairs:
                stack.append(pairs[curr])
            elif curr in closers:
                if not stack or curr != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    yield text[start:idx + 1].strip()
                    break
            idx += 1

def _json_parse_candidates(text: str):
    stripped = text.strip()
    candidates = [_strip_markdown_code_fence(stripped)]
    candidates.extend(_markdown_json_blocks(stripped))
    candidates.extend(_balanced_json_spans(stripped))

    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate

def _quote_bare_json5_values(text: str) -> str:
    """
    Repair a common LLM JSON mistake: object values written as bare prose.

    Example:
        "waypoints": Waypoints 1-13 are visible.
    becomes:
        "waypoints": "Waypoints 1-13 are visible."

    This intentionally works line-by-line and only targets values after a colon.
    Valid JSON/JSON5 starts such as strings, objects, arrays, numbers and literals
    are left untouched.
    """
    valid_bare_literals = {"true", "false", "null", "nan", "infinity", "+infinity", "-infinity"}
    repaired_lines = []

    for line in text.splitlines():
        match = re.match(r'^(\s*(?:(?:"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][\w$-]*)\s*:\s*))(.*?)(\s*,?\s*)$', line)
        if not match:
            repaired_lines.append(line)
            continue

        prefix, raw_value, suffix = match.groups()
        value = raw_value.strip()
        if not value:
            repaired_lines.append(line)
            continue

        first_char = value[0]
        lower_value = value.lower()
        is_valid_start = (
            first_char in ('"', "'", "{", "[")
            or first_char.isdigit()
            or first_char in ("-", "+", ".")
            or lower_value in valid_bare_literals
        )
        if is_valid_start:
            repaired_lines.append(line)
            continue

        repaired_lines.append(f"{prefix}{json.dumps(value, ensure_ascii=False)}{suffix}")

    return "\n".join(repaired_lines)

def robust_json_parse(json_str: str):
    """
    Parse a JSON string using the json5 library.
    Supports trailing commas, comments, single quotes, etc.
    Also handles mixed JSON and plain text in LLM output; auto-extracts first parseable fragment.
    """
    if not json_str:
        return None

    last_error = None
    for candidate in _json_parse_candidates(str(json_str)):
        try:

            return json5.loads(candidate)
        except Exception as e:
            last_error = e
            try:
                return json5.loads(_quote_bare_json5_values(candidate))
            except Exception as repaired_e:
                last_error = repaired_e

    print(f"JSON5 parse failed: {last_error}")
    return None
