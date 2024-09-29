from ultralytics import YOLO

model = YOLO('yolov8x.pt')

results = model.train(data='data/data.yaml',
                      epochs=50,
                      imgsz=640,
                      batch=10,
                      plots=True)