from ultralytics import YOLO

# تحميل النموذج المدرب
model = YOLO("/HandDetect/Models/best.pt")

# التحقق على بيانات validation
metrics = model.val(data="C:/Users/ragad/PycharmProjects/Yolov5HandDectetion/HandDetect/hands.v2i.yolov8-obb/data.yaml", split="val")

# عرض النتائج
print(metrics)
