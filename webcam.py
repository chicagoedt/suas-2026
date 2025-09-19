from ultralytics import YOLO
import cv2 as cv

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize webcam
cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    image = results[0].plot()
    cv.imshow('results', image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()