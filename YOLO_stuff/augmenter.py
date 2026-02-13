import cv2
import numpy as np
import math
import os
from pathlib import Path

# --- CONFIG ---
INPUT_IMAGE = "y/images/IMG_E7868.jpg"
INPUT_LABEL = "y/labels/IMG_E7868.txt"
OUTPUT_IMAGES_DIR = "train/images"
OUTPUT_LABELS_DIR = "train/labels"

AUGMENTATIONS = {
    "+28": ("rotate", -28),
    "-35": ("rotate", -35),
    "+60": ("rotate", 60),
    "-60": ("rotate", -60),
    "+10": ("rotate", 10),
    "-67": ("rotate", -67),
    "+93": ("rotate", 93),
    "-99": ("rotate", -99),
    "-120": ("rotate", -120),
    "+120": ("rotate", 120),
    "-150": ("rotate", -150),
    "+150": ("rotate", 150),
    "-116": ("rotate", -116),
    "+180": ("rotate", 180),
    "skew_fwd+15": ("skew", 15),
    "skew_fwd+40": ("skew", 40),
    "skew_back-15": ("skew", -15),
    "skew_back-30": ("skew", -30),
    "skew_back-40": ("skew", -40),
    "skew_back-60": ("skew", -60)
}
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# --- Helper Functions ---
def read_yolo_polygon_labels(label_path, img_w, img_h):
    """
    YOLO polygon format:
    class_id x1 y1 x2 y2 ... (normalized)
    Returns list of tuples: (cls, Nx2 array of pixel coords)
    """
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            labels.append((cls, coords))
    return labels

def write_yolo_polygon_labels(label_path, boxes, img_w, img_h):
    """
    Writes YOLOv11 polygon annotation: class_id x1 y1 x2 y2 ... (normalized)
    """
    with open(label_path, "w") as f:
        for cls, poly in boxes:
            norm = poly.copy()
            norm[:, 0] /= img_w
            norm[:, 1] /= img_h
            flat = " ".join(map(lambda x: f"{x:.6f}", norm.flatten()))
            f.write(f"{cls} {flat}\n")

def rotate90(image, polys, direction=True):
    """
    Rotate an image and its polygons by 90 degrees.

    Args:
        image: HxWxC uint8 np.ndarray
        polys: list of (cls, Nx2) polygons
        direction: True for clockwise (CW), False for counter-clockwise (CCW)
    Returns:
        rotated_image, rotated_polys
    """
    h, w = image.shape[:2]

    if direction:  # Clockwise
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        new_polys = []
        for cls, pts in polys:
            # (x, y) -> (y, w - 1 - x)
            new_pts = np.zeros_like(pts)
            new_pts[:, 0] = pts[:, 1]
            new_pts[:, 1] = w - 1 - pts[:, 0]
            new_polys.append((cls, new_pts))
    else:  # Counter-clockwise
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_polys = []
        for cls, pts in polys:
            # (x, y) -> (h - 1 - y, x)
            new_pts = np.zeros_like(pts)
            new_pts[:, 0] = h - 1 - pts[:, 1]
            new_pts[:, 1] = pts[:, 0]
            new_polys.append((cls, new_pts))

    return rotated_image, new_polys


def rotate180(image, polys):
    """
    Rotate an image and its polygons by 180 degrees (flip upside down).

    Args:
        image: HxWxC uint8 np.ndarray
        polys: list of (cls, Nx2) polygons
    Returns:
        rotated_image, rotated_polys
    """
    h, w = image.shape[:2]
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    new_polys = []
    for cls, pts in polys:
        # (x, y) -> (w - 1 - x, h - 1 - y)
        new_pts = np.zeros_like(pts)
        new_pts[:, 0] = w - 1 - pts[:, 0]
        new_pts[:, 1] = h - 1 - pts[:, 1]
        new_polys.append((cls, new_pts))

    return rotated_image, new_polys

def rotate_image_and_polygons(image, polys, angle = 45):
    """
    Rotates an image and polygon annotations by a given angle (degrees).
    Supports any angle, including beyond ±90°, by handling 90° multiples cleanly.
    """
    print("rotate_image_and_polygons() angle", angle)
    angle = angle % 360
    if angle > 180:
        angle -= 360
    if angle < -180:
        angle += 360

    # --- Handle exact 90/180 multiples directly ---
    if abs(angle) == 90:
        return rotate90(image, polys, direction=(angle > 0))
    elif abs(angle) == 180:
        return rotate180(image, polys)
    elif abs(angle) == 270:
        return rotate90(image, polys, direction=(angle < 0))
    elif angle == 0:
        return image.copy(), [(cls, pts.copy()) for cls, pts in polys]

    # --- Handle combined rotation ---
    # Separate 90° multiples from remainder
    num_quads = int(np.floor(angle / 90.0))
    remainder = angle - num_quads * 90

    # Perform the integer-multiple 90° rotations first
    rotated_img = image.copy()
    rotated_polys = polys.copy()
    for _ in range(abs(num_quads)):
        rotated_img, rotated_polys = rotate90(rotated_img, rotated_polys, direction=(num_quads > 0))

    (h, w) = image.shape[:2]
    if angle != 0:
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(rot_mat[0, 0]), np.abs(rot_mat[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=(128,128,128))

        new_polys = []
        for cls, pts in polys:
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
            new_pts = rot_mat.dot(pts_h.T).T
            new_polys.append((cls, new_pts))

    crop_x, crop_y = crop_after_rotation_helper(w, h, angle)

    cropped_img, cropped_polys = crop_image_and_polygons(
        rotated, new_polys,
        left=crop_x,
        right=crop_x,
        top=crop_y,
        bottom=crop_y
    )

    return cropped_img, cropped_polys

def crop_after_rotation_helper(w, h, theta):
    """
    Compute how many pixels to crop from each side after rotating an image by theta_deg degrees,
    keeping the same aspect ratio and centering the crop (like iOS Photos rotation).

    Args:
        w (float): original image width in pixels
        h (float): original image height in pixels
        theta_deg (float): rotation angle in degrees (|theta| < 45° recommended)

    Returns:
        dict: {
            "crop_left":  pixels to remove from left,
            "crop_right": pixels to remove from right,
            "crop_top":   pixels to remove from top,
            "crop_bottom":pixels to remove from bottom,
            "new_width":  resulting cropped width,
            "new_height": resulting cropped height,
            "scale":      final crop/original size ratio
        }
    """
    theta = math.radians(theta)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))

    # Two possible scale limits
    s1 = w / (w * cos_t + h * sin_t)
    s2 = h / (w * sin_t + h * cos_t)

    # Choose the limiting scale
    s = min(s1, s2)

    # New cropped dimensions
    new_w = s * w
    new_h = s * h

    # Equal crop on both sides (centered)
    crop_x = (w - new_w) / 1
    crop_y = (h - new_h) / 1

    # print("crop_x:", crop_x)
    # print("crop_y", crop_y)

    return int(crop_x), int(crop_y)

def skew_image_and_polygons_yaxis(image, polys, angle_deg=30):
    """Simulate Y-axis rotation (keystone skew) and crop to valid pixels."""
    h, w = image.shape[:2]
    angle = np.radians(angle_deg)
    
    # Compute depth offset (horizontal compression)
    offset = int((w / 2) * np.tan(angle) * 0.5)  # smaller factor prevents over-skewing
    
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    if angle_deg >= 0:
        # Top compressed inward
        dst = np.float32([
            [offset, 0],
            [w - offset, 0],
            [0, h],
            [w, h]
        ])
    else:  # backward (bottom compressed inward)
        angle_deg *= -1
        dst = np.float32([
            [0, 0],
            [w, 0],
            [offset, h],
            [w - offset, h]
        ])

    # Compute perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), borderValue=(0, 0, 0))

    # Transform polygons using same matrix
    new_polys = []
    for cls, pts in polys:
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        new_pts = (M @ pts_h.T).T
        new_pts = new_pts[:, :2] / new_pts[:, 2, np.newaxis]
        new_polys.append((cls, new_pts))

    # Compute mask to find valid region (non-black pixels)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask = (gray > 5).astype(np.uint8)
    x, y, ww, hh = cv2.boundingRect(mask)

    # Crop to the actual visible pixels
    cropped_img, cropped_polys = crop_image_and_polygons(
        warped, new_polys,
        left=offset,
        right=offset,
        top=0, bottom=0
    )

    return cropped_img, cropped_polys

def crop_image_and_polygons(image, polys, left=0, right=0, top=0, bottom=0):
    """
    Crop image by the given pixel margins on each side,
    and transform the polygon coordinates accordingly.

    Args:
        image (np.ndarray): The input image (H, W, 3)
        polys (list): List of tuples (class_id, Nx2 polygon array)
        left, right, top, bottom (int): Crop margins in pixels

    Returns:
        cropped_image (np.ndarray)
        cropped_polys (list of (class_id, Nx2))
    """

    h, w = image.shape[:2]

    # print("h:", h)
    # print("w:", w)
    # print("left:", left)
    # print("right:", right)
    # print("top:", top)
    # print("bottom:", bottom)

    # clamp values so we don't crop past the image edges
    left = max(0, min(left, w - 1))
    right = max(0, min(right, w - left - 1))
    top = max(0, min(top, h - 1))
    bottom = max(0, min(bottom, h - top - 1))

    # compute new bounds
    x_start = left
    x_end = w - right
    y_start = top
    y_end = h - bottom

    cropped_image = image[y_start:y_end, x_start:x_end]

    # adjust polygons
    new_polys = []
    for cls, pts in polys:
        new_pts = pts.copy()
        # shift coordinates to match crop origin
        new_pts[:, 0] -= x_start
        new_pts[:, 1] -= y_start

        # clip coordinates to cropped image size
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, cropped_image.shape[1] - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, cropped_image.shape[0] - 1)

        new_polys.append((cls, new_pts))

    return cropped_image, new_polys

# --- Main ---
image = cv2.imread(INPUT_IMAGE)
if image is None:
    raise FileNotFoundError(f"Cannot read image: {INPUT_IMAGE}")

h, w = image.shape[:2]
polygons = read_yolo_polygon_labels(INPUT_LABEL, w, h)

base_name = Path(INPUT_IMAGE).stem

for suffix, (aug_type, param) in AUGMENTATIONS.items():
    if aug_type == "rotate":
        print("rotate", param)
        aug_img, aug_polys = rotate_image_and_polygons(image, polygons, param)
    elif aug_type == "skew":
        print("skew", param)
        aug_img, aug_polys = skew_image_and_polygons_yaxis(image, polygons, param)
    else:
        continue

    # Save image
    out_img_name = f"{base_name}_{suffix}.jpg"
    out_img_path = os.path.join(OUTPUT_IMAGES_DIR, out_img_name)
    cv2.imwrite(out_img_path, aug_img)

    # Save labels
    out_lbl_name = f"{base_name}_{suffix}.txt"
    out_lbl_path = os.path.join(OUTPUT_LABELS_DIR, out_lbl_name)
    write_yolo_polygon_labels(out_lbl_path, aug_polys, aug_img.shape[1], aug_img.shape[0])

print("✅ Done. Augmented images saved to /images and labels to /labels.")