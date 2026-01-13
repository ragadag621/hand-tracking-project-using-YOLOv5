
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("C:/Users/ragad/PycharmProjects/Yolov5HandDectetion/HandDetect/runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    hsv = None
    mask = None
    for result in results:
        if result.boxes:
            for box in result.boxes:
                if len(box.xyxy[0]) == 4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                elif len(box.xyxy[0]) == 5:
                    x1, y1, x2, y2, confidence = map(int, box.xyxy[0])
                else:
                    continue

                if confidence > 0.5:
                    # رسم المربع
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # حساب منتصف المربع
                    centerX = (x1 + x2) // 2
                    centerY = (y1 + y2) // 2

                    # رسم النقطة في منتصف المربع
                    cv2.circle(frame, (centerX, centerY), 2, (0, 255, 0), -1)

            hand_roi = frame[y1:y2, x1:x2]

            hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

    cv2.imshow("Original Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
