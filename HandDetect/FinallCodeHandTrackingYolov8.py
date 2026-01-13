#!/usr/bin/env python3

# Import necessary libraries
import enum
import sys

# Append custom library path
sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *
import cv2
import ipywidgets.widgets as widgets
import threading
import time
import math
import PID
import inspect
import ctypes
import numpy as np
from ultralytics import YOLO

# Global variables for hand detection coordinates
global hand_x, hand_y, hand_w, hand_h
hand_x = hand_y = hand_w = hand_h = 0

global target_valuex
target_valuex = 2048

# Load the YOLO model for hand detection
model = YOLO("/best.pt")


# Function to convert BGR image to JPEG
def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])


# Function to forcefully raise an exception in a thread
def _async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("Invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)


# Function to stop a thread safely
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


# Initialize PID controllers for movement
direction_pid = PID.PositionalPID(0.8, 0, 0.2)
yservo_pid = PID.PositionalPID(0.8, 0.2, 0.01)
speed_pid = PID.PositionalPID(1.1, 0, 0.2)


# Reset servos to their default positions
def servo_reset():
    bot.Ctrl_Servo(1, 90)
    bot.Ctrl_Servo(2, 80)


# Reset coordinate system and PID outputs
def reset_coordinate_system():
    global x, w, y, h
    x, w, y, h = 0, 0, 0, 0
    direction_pid.SystemOutput = 0
    yservo_pid.SystemOutput = 0


# Function to filter skin color in the frame
def filter_skin(frame):
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Lower HSV range for skin
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)  # Upper HSV range for skin

    # Create a binary mask for skin color
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply the mask to the original frame
    skin_filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return skin_filtered_frame


# Search for a hand by sweeping the servo and analyzing video frames
def search_for_hand():
    found_hand = False
    angle = 90
    step = 10
    max_angle = 180
    min_angle = 0

    while not found_hand:
        bot.Ctrl_Servo(1, angle)
        time.sleep(0.2)
        ret, frame = image.read()
        if not ret or frame is None:
            continue

        # Apply skin filter
        skin_filtered_frame = filter_skin(frame)

        cv2.imshow("Searching for Hand", skin_filtered_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        results = model(skin_filtered_frame)
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    if len(box.xyxy[0]) >= 4:
                        confidence = box.conf[0]
                        if confidence > 0.6:
                            found_hand = True
                            bot.Ctrl_Servo(1, 90)
                            return True

        angle += step
        if angle >= max_angle or angle <= min_angle:
            step = -step

    return False


# Follow a detected hand and control the robot's movement
def Hand_Follow():
    global x, w, y, h
    speed = 30
    reference_area = None

    if not search_for_hand():
        stop_robot()
        return

    while True:
        ret, frame = image.read()
        if not ret or frame is None:
            stop_robot()
            continue

        # Apply skin filter
        skin_filtered_frame = filter_skin(frame)

        results = model(skin_filtered_frame)
        bbox = None
        center_x, center_y = 0, 0

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    if len(box.xyxy[0]) >= 4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        if confidence > 0.6:
                            bbox = (x1, y1, x2 - x1, y2 - y1)
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), -1)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if bbox:
            x, y, w, h = bbox
            current_area = w * h
            if reference_area is None:
                reference_area = current_area
                continue

            direction_pid.SystemOutput = center_x
            direction_pid.SetStepSignal(int(image_width / 2))
            direction_pid.SetInertiaTime(0.01, 0.05)
            target_valuex = int(direction_pid.SystemOutput)

            if current_area < reference_area * 1.5:
                move_forward(speed)
            elif current_area > reference_area * 3:
                move_backward(speed)
            else:
                stop_robot()


# Main function to initialize components and handle hand tracking
def main():
    global image, image_widget, image_width, image_height

    servo_reset()

    image = cv2.VideoCapture(0)
    image_width = 640
    image_height = 480
    image.set(3, image_width)
    image.set(4, image_height)
    image_width = image.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = image.get(cv2.CAP_PROP_FRAME_HEIGHT)

    image_widget = widgets.Image(format='jpeg', width=640, height=480)

    thread1 = threading.Thread(target=Hand_Follow)
    thread1.daemon = True
    thread1.start()

    try:
        while thread1.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_thread(thread1)
        image.release()
        cv2.destroyAllWindows()
        servo_reset()


if __name__ == "__main__":
    main()
