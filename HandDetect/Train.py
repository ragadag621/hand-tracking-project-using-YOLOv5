from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='C:/Users/ragad/PycharmProjects/Yolov5HandDectetion/HandDetect/hands.v2i.yolov8-obb/data.yaml', epochs=200, imgsz=640)
