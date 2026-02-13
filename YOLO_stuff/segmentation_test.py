import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test1.jpg")

for result in results:

    img = result.orig_img.copy()

    if result.masks is None:
        continue

    masks = result.masks.data.cpu().numpy()

    for mask in masks:

        # ---- Resize mask to original image size ----
        mask_resized = cv2.resize(
            mask,
            (img.shape[1], img.shape[0]),
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

        # Polygon simplification
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        MAX_VERTICES = 8
        while len(approx) > MAX_VERTICES:
            epsilon *= 1.2
            approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw polygon
        cv2.polylines(img, [approx], True, (0, 255, 0), 2)

        for pt in approx:
            x, y = pt[0]
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

    display_img = cv2.resize(img,(img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)

    cv2.imshow("Simplified Polygon", display_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
