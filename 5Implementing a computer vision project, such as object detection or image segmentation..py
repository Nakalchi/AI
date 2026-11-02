import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

print("Loading YOLOv5 model...")

model = YOLO(r"C:\Users\Mandar\Downloads\yolov5s.pt")

image_path = r"C:\Users\Mandar\Downloads\ash.webp"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Performing object detection...")

results = model.predict(image_rgb, conf=0.5)

annotated_image = results[0].plot()

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Object Detection Results")
plt.show()

output_path = r"C:\Users\Mandar\Downloads\output.jpg"

cv2.imwrite(output_path, annotated_image)

print("Annotated image saved to:", output_path)

print("\nDetected objects:")

for box in results[0].boxes.data.tolist():

    x1, y1, x2, y2, conf, cls = box

    print(
        f"Class: {results[0].names[int(cls)]}, "
        f"Confidence: {conf:.2f}, "
        f"Coordinates: ({x1:.2f}, {y1:.2f}), "
        f"({x2:.2f}, {y2:.2f})"
    )
