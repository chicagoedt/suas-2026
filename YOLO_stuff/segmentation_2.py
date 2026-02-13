import cv2
import numpy as np
from ultralytics import YOLO


# ---------- Helper: order corners ----------
def order_points(pts):
    pts = pts.reshape(4, 2)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )

def order_points_angle(pts):
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


# ---------- Load model ----------
model = YOLO("best.pt")

results = model("test2_upsidedown.jpg")

for result in results:

    img = result.orig_img.copy()

    if result.masks is None:
        continue

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    deskewed_img = None

    for mask, cls in zip(masks, classes):

        # Resize mask to original image size
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

        # ---- Simplify polygon ----
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        MAX_VERTICES = 4
        while len(approx) > MAX_VERTICES:
            epsilon *= 1.2
            approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw simplified polygon
        cv2.polylines(img, [approx], True, (0, 255, 0), 2)

        for pt in approx:
            x, y = pt[0]
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        # ---- Deskew only class "gcp" (index 0) ----
        class_name = result.names[cls]
        print(class_name, "vertices:", len(approx))
        if class_name == "gcp" and len(approx) == 4:

            src_pts = order_points_angle(approx)

            #  Draw ordered points

            for i, p in enumerate(src_pts.astype(int)):
                cv2.putText(img, str(i), tuple(p),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255,0,0), 2)

            # Compute square size from detected edges
            width = int(max(
                np.linalg.norm(src_pts[0] - src_pts[1]),
                np.linalg.norm(src_pts[2] - src_pts[3])
            ))

            height = int(max(
                np.linalg.norm(src_pts[1] - src_pts[2]),
                np.linalg.norm(src_pts[3] - src_pts[0])
            ))

            output_size = max(width, height)

            dst_pts = np.array([
                [0, 0],
                [output_size - 1, 0],
                [output_size - 1, output_size - 1],
                [0, output_size - 1]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            deskewed_img = cv2.warpPerspective(
                result.orig_img,
                M,
                (output_size, output_size)
            )

            hsv = cv2.cvtColor(deskewed_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            c = output_size // 2
            
            mask0 = np.zeros(v.shape, dtype=np.uint8)
            p0 = np.array([[0, 0], [c, c], [output_size, 0]])
            cv2.fillPoly(mask0, [p0], 255)
            mask1 = np.zeros(v.shape, dtype=np.uint8)
            p1 = np.array([[output_size, 0], [c, c], [output_size, output_size]])
            cv2.fillPoly(mask1, [p1], 255)
            mask2 = np.zeros(v.shape, dtype=np.uint8)
            p2 = np.array([[0, output_size], [c, c], [output_size, output_size]])
            cv2.fillPoly(mask2, [p2], 255)
            mask3 = np.zeros(v.shape, dtype=np.uint8)
            p3 = np.array([[0, 0], [c, c], [0, output_size]])
            cv2.fillPoly(mask3, [p3], 255)

            avg_b = {0: cv2.mean(v, mask=mask0)[0], 1: cv2.mean(v, mask=mask1)[0], 2: cv2.mean(v, mask=mask2)[0], 3: cv2.mean(v, mask=mask3)[0]}
##            avg_p0 = cv2.mean(v, mask=mask0)[0]
##            avg_p1 = cv2.mean(v, mask=mask1)[0]
##            avg_p2 = cv2.mean(v, mask=mask2)[0]
##            avg_p3 = cv2.mean(v, mask=mask3)[0]

##            brightnesses = {}
##
##            for i in sorted_avg_b:
##                brightnesses[i] 
##
##            print("p0", avg_p0)
##            print("p1", avg_p1)
##            print("p2", avg_p2)
##            print("p3", avg_p3)

            sorted_avg_b = {k: v for k, v in sorted(avg_b.items(), key=lambda item: item[1])}
            print(sorted_avg_b)
            i_num = list(sorted_avg_b.keys())[2]
            print("2nd brightest:" + str(list(sorted_avg_b.keys())[2]))

            if(i_num == 0): 
                deskewed_img_rotated = cv2.rotate(deskewed_img, cv2.ROTATE_90_CLOCKWISE)
            elif(i_num == 2): 
                deskewed_img_rotated = cv2.rotate(deskewed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif(i_num == 3): 
                deskewed_img_rotated = cv2.rotate(deskewed_img, cv2.ROTATE_180)

            cv2.imshow("Rotated GCP", deskewed_img_rotated)

            results2 = model(deskewed_img_rotated)
            for r in results2:
                for cls_id in r.boxes.cls:
                    print("Detected: " + model.names[int(cls_id)])

    # ---- Display highlighted image (half size) ----
    display_img = cv2.resize(
        img,
        (img.shape[1] // 2, img.shape[0] // 2),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Highlighted", display_img)

    if deskewed_img is not None:
        print("deskewed_img is not None")
        cv2.imshow("Deskewed GCP", deskewed_img)

    cv2.waitKey(0)

cv2.destroyAllWindows()
