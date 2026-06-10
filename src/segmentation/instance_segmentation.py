import numpy as np
from torch.backends.mkl import verbose
from ultralytics import YOLOE
import cv2
from segmentation.object_list import *
import os
import matplotlib.pyplot as plt


# load model once globally
MODEL_PATH = "../ckpt/yoloe-26l-seg.pt"
model = YOLOE(MODEL_PATH)
category_names = {cat['id']: cat['name'] for cat in categories}
model.set_classes(list(category_names.values()), model.get_text_pe(list(category_names.values())))

def instance_segmentation(image: np.ndarray):
    """
    Input: image (np.ndarray) - raw image
    Output: List[dict] - per-instance class_id, class_name, mask (same size as input)
    """
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    edge_threshold_x = w * 0.2
    edge_threshold_y = h * 0.2

    results = model.predict(image, conf=0.6, iou=0.3, retina_masks=True, verbose=False)
    result = results[0]
    output = []
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        class_names = result.names
        for i in range(masks.shape[0]):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if (center_x < edge_threshold_x or center_y < edge_threshold_y or
                center_x > w - edge_threshold_x or center_y > h - edge_threshold_y):
                continue

            class_id = int(boxes.cls[i].item())
            class_name = class_names.get(class_id, str(class_id))
            mask = masks[i].astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            output.append({
                "class_id": class_id,
                "class_name": class_name,
                "mask": mask_resized
            })
    return output

def get_class_color(class_id: int) -> tuple:
    # use matplotlib tab20 colormap for fixed, distinct per-class colors
    color_map = plt.get_cmap('tab20')
    color = color_map(class_id % 20)[:3]  # take first 3 channels (RGB)
    return tuple(int(255 * c) for c in color)

def visualize_instance_segmentation(image: np.ndarray, instances: list, alpha: float = 0.5) -> np.ndarray:
    vis_image = image.copy()
    if vis_image.shape[2] == 4:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2BGR)
    for inst in instances:
        class_id = inst["class_id"]
        color = get_class_color(class_id)
        mask = inst["mask"]
        colored_mask = np.zeros_like(vis_image)
        for c in range(3):
            colored_mask[:, :, c] = mask // 255 * color[c]
        vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            center_x, center_y = int(xs.mean()), int(ys.mean())
            center_x = np.clip(center_x, 0, vis_image.shape[1] - 1)
            center_y = np.clip(center_y, 0, vis_image.shape[0] - 1)
            cv2.putText(vis_image, inst["class_name"], (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(vis_image, inst["class_name"], (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return vis_image