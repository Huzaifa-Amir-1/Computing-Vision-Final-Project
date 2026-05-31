import sys

import cv2

from inference import load_detection_model, predict_image

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
image = cv2.imread(image_path)

if image is None:
    print(f"Could not read image: {image_path}")
    sys.exit(1)

model = load_detection_model()
result = predict_image(model, image)

print(f"{result['label']} Image ({result['confidence'] * 100:.1f}% confidence)")
print(f"Fake probability: {result['fake_probability'] * 100:.1f}%")
