from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')

predictions = model.predict(source='runs/prediction/896_1152_2560_2816_png_jpg.rf.54b1752262847cc169f62451f9d2c063.jpg', # path to your image
                            save=True,
                            show_labels=False,
                            project="runs",
                            conf=0.2)
