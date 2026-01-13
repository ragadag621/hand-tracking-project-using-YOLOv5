from ultralytics import YOLO

# تحميل النموذج المدرب
model = YOLO("C:/Users/ragad/PycharmProjects/Yolov5HandDectetion/HandDetect/runs/detect/train3/weights/best.pt")

# اختبار النموذج على بيانات test
test_metrics = model.val(data="C:/Users/ragad/PycharmProjects/Yolov5HandDectetion/HandDetect/hands.v2i.yolov8-obb/data.yaml", split="test")

# عرض النتائج
print(test_metrics)
