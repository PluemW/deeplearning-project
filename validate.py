from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/best.pt')

results = model.val(data='data/data.yaml')