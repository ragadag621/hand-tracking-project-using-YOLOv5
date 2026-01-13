# 🏎️ Autonomous Hand-Following Robot | Raspberry Pi 5 & Edge AI

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red)
![AI-Model](https://img.shields.io/badge/AI-YOLOv8-blue)
![Computer-Vision](https://img.shields.io/badge/Vision-OpenCV%20%2B%20HSV-orange)

An advanced robotic system that integrates **Deep Learning** with **Classical Computer Vision** to track and follow human hand movements in real-time. This project isn't just a simple detection script; it's a full-cycle engineering effort from data curation to hardware deployment on the latest **Raspberry Pi 5**.

---

## 🏗️ Project Lifecycle & Engineering Effort

This project was built through a rigorous 4-stage engineering process:

### 1. Data Engineering (Collection & Pre-processing)
* **Custom Dataset Curation:** I collected and labeled a specific dataset for hand gestures to ensure the model performs under various lighting and backgrounds.
* **Data Augmentation:** Applied techniques like rotation, scaling, and brightness adjustment to make the model robust against environmental noise.
* **Class Balancing:** Focused on diverse hand orientations and distances to avoid detection bias.

### 2. Model Training (YOLOv8 Optimization)
* **Architecture:** Leveraged **YOLOv8 (Nano)** for its superior speed-to-accuracy ratio on edge devices.
* **Training:** Conducted multiple training sessions (`train1`, `train2`, `train3`) to fine-tune hyperparameters for real-time inference.
* **Weights:** Exported the optimized `best.pt` for deployment, achieving high mAP (mean Average Precision).

### 3. The Vision Pipeline (Hybrid Approach)
To achieve industrial-grade reliability, I implemented a **Dual-Verification System**:
* **Deep Learning Layer:** YOLOv8 identifies the hand's global Bounding Box.
* **Traditional CV Layer:** Inside the ROI (Region of Interest), the system applies **HSV Color Segmentation** to lock onto human skin tones, preventing the robot from following non-human objects.
* **Morphological Refining:** Used `Erode`, `Dilate`, and `Gaussian Blur` to eliminate sensor noise and flickering for a cleaner tracking signal.

### 4. Control Theory (PID & Robotics)
* **Centroid Tracking:** Real-time calculation of the hand's center point relative to the camera's optical center.
* **PID Controller:** Implemented a **Proportional-Integral-Derivative** controller to ensure the robot's movement is smooth, preventing jerky starts and stops.
* **Distance Logic:** Dynamically estimated distance based on the **Bounding Box Area**. The robot advances as the area decreases (target is far) and retreats if it exceeds a safety threshold (target is too close).

---

## 🛠️ Tech Stack & Components

* **Brain:** Raspberry Pi 5 (8GB RAM)
* **AI Framework:** Ultralytics YOLOv8, PyTorch
* **Vision Library:** OpenCV (Python)
* **Controller:** Custom PID Feedback Loop
* **Hardware:** McLumk Wheel Sports Chassis, Dual-axis Servos, USB Camera.

---

## 📂 Repository Structure

* `HandDetect/`: Core package containing the inference engine.
* `HandReco.py`: Vision testing module for camera and HSV calibration.
* `FinallCodeHandTrackingYolov8.py`: The production-ready autonomous robot control script.
* `Models/`: Contains the optimized `best.pt` and `last.pt` weight files.
* `lib/`: Low-level hardware drivers for the Raspberry Pi 5 motor controller.

---

## 📺 Live Demonstration
Check out the robot in action following my hand gestures in real-time:



> [!IMPORTANT]
> **View the Video:** https://s2.ezgif.com/tmp/ezgif-22981b79f7255f5f.gif

---

## 🏁 How to Run
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/ragadag621/hand-tracking-project-using-YOLOv5.git](https://github.com/ragadag621/hand-tracking-project-using-YOLOv5.git)
