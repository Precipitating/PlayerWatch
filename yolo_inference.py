from ultralytics import YOLO

# load model
model = YOLO('models/best.pt')

results = model.predict('input_videos/spursy.mp4', save=True)

for box in results[0].boxes:
    print(box)