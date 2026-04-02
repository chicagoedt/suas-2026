import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def wait_for_windows_to_close(window_names, poll_ms=20):
    while True:
        key = cv2.waitKey(poll_ms) & 0xFF
        if key in (27, ord("q")):
            break

        open_windows = 0
        for name in window_names:
            try:
                if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
                    open_windows += 1
            except cv2.error:
                continue

        if open_windows == 0:
            break

# orders corners of ground control point
# returns an array in order:
#   1------2
#   |      |
#   |      |
#   3------4
def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    c = pts.mean(axis=0)

    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
    pts = pts[np.argsort(angles)]  # CW order around centroid

    # rotate so index 0 is top-left (min x+y)
    start = np.argmin(pts.sum(axis=1))
    pts = np.roll(pts, -start, axis=0)

    # ensure order is TL, TR, BR, BL (clockwise)
    # If it's CCW, swap TR and BL
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    if cross(pts[0], pts[1], pts[2]) < 0:  # CW in image coords (y down) can be tricky
        pts[[1,3]] = pts[[3,1]]

    return pts

# ---------- Helper ----------
# Rotates image using brightness of triangular quadrants of square image
# Second brightest triangle should be the triangle with the number on it
# img: deskewed image of ground control point
# debug: optional parameter to enable debugging outputs
def rotate_img(img, output_size, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    
    triangles = square_quadrant_triangles(output_size)
    masks = [triangle_mask(v.shape, t) for t in triangles]

    avg_brightness = average_brightness_per_mask(v, masks)
    sorted_brightness = sorted(avg_brightness.items(), key=lambda x: x[1])

    i_num = sorted_brightness[2][0]

    if debug:
        print("Brightness per quadrant: ", avg_brightness)
        print("Second brightest: ", i_num)

    rotations = {
        0: cv2.ROTATE_90_CLOCKWISE,
        2: cv2.ROTATE_90_COUNTERCLOCKWISE,
        3: cv2.ROTATE_180
    }

    return cv2.rotate(img, rotations[i_num]) if i_num in rotations else img

def compute_output_size(src_pts):
    width = int(max(
        np.linalg.norm(src_pts[0] - src_pts[1]),
        np.linalg.norm(src_pts[2] - src_pts[3])
    ))

    height = int(max(
        np.linalg.norm(src_pts[1] - src_pts[2]),
        np.linalg.norm(src_pts[3] - src_pts[0])
    ))

    return max(width, height)

def draw_points(img, points, color=(255, 0, 0)):
    img = img.copy()
    for i, p in enumerate(points.astype(int)):
        cv2.putText(img, str(i), tuple(p),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,0,0), 2)
    return img

def deskew_and_rotate_img(vertices, img, debug=False, debug_prefix="", shown_windows=None):
    src_pts = order_points(vertices)
    output_size = compute_output_size(src_pts)

    #  Draw ordered points
    # for i, p in enumerate(src_pts.astype(int)):
    #     cv2.putText(img, str(i), tuple(p),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (255,0,0), 2)


    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    deskewed_img = cv2.warpPerspective(
        img,
        M,
        (output_size, output_size)
    )

    if debug is True and deskewed_img is not None:
        print("deskewed_img is not None")
        cv2.imshow("Deskewed GCP", deskewed_img)
        if shown_windows is not None:
            shown_windows.add("Deskewed GCP")
    cv2.imwrite(f"segmentation_debug/{debug_prefix}deskewed.jpg", deskewed_img)

    return rotate_img(deskewed_img, output_size)

def display_highlighted_img(img):
    display_img = cv2.resize(
            img,
            (img.shape[1] // 2, img.shape[0] // 2),
            interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Highlighted", display_img)
    return "Highlighted"

def triangle_mask(shape, vertices):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [vertices.astype(np.int32)], 255)
    return mask

def square_quadrant_triangles(size):
    c = size // 2
    return [
        np.array([[0, 0], [c, c], [size, 0]]),          # top
        np.array([[size, 0], [c, c], [size, size]]),   # right
        np.array([[0, size], [c, c], [size, size]]),   # bottom
        np.array([[0, 0], [c, c], [0, size]])           # left
    ]

def average_brightness_per_mask(channel, masks):
    return {
        i: cv2.mean(channel, mask=m)[0]
        for i, m in enumerate(masks)
    }

def main():

    filename = "unclear_test.jpg"
    filename_stem = Path(filename).stem

    # ---------- Load model ----------
    model = YOLO("best.pt")

    results = model(filename)

    for result in results:
        shown_windows = set()

        if result.masks is None:
            continue
        
        # Clean layer
        orig = result.orig_img.copy()

        # Visualization layer
        vis = orig.copy()

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for mask, cls in zip(masks, classes):

            # Resize mask to original image size
            mask_resized = cv2.resize(
                mask,
                (orig.shape[1], orig.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)

            # ---- Simplify polygon ----
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True) 

            while len(approx) > 4:
                epsilon *= 1.2
                approx = cv2.approxPolyDP(contour, epsilon, True)

            # Draw simplified polygon
            #cv2.polylines(vis, [approx], True, (0, 255, 0), 2)

            for pt in approx:
                x, y = pt[0]
                cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

            class_name = result.names[cls]
            print(f"{class_name} vertices: {len(approx)}")

            # ---- Deskew + Rotate only GCP ----
            if class_name == "gcp" and len(approx) == 4:

                rotated_gcp = deskew_and_rotate_img(
                    approx,
                    orig,
                    debug_prefix=f"{filename_stem}_",
                    shown_windows=shown_windows,
                )
                cv2.imwrite(
                    f"segmentation_debug/{filename_stem}_rotated_gcp.jpg",
                    rotated_gcp
                )

                if rotated_gcp is None:
                    continue
                
                cv2.imshow("Rotated GCP", rotated_gcp)
                shown_windows.add("Rotated GCP")

                results2 = model(rotated_gcp)

                for r in results2:
                    if r.boxes is None:
                        continue
                    for cls_id in r.boxes.cls:
                        print("Detected: " + model.names[int(cls_id)])

        highlighted_window = display_highlighted_img(vis)
        shown_windows.add(highlighted_window)

        wait_for_windows_to_close(shown_windows)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

main()
