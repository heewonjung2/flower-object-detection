from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config():
    """Load project configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def predict_images():
    """Detect flowers in multiple images and visualize results using OpenCV."""
    config = load_config()

    confidence_threshold = config["prediction"]["confidence_threshold"]

    # Path settings
    model_path = (
        BASE_DIR
        / "results"
        / "flower_detection"
        / "weights"
        / "best.pt"
    )

    input_dir = BASE_DIR / "data" / "test_images"
    output_dir = BASE_DIR / "results" / "opencv_predictions"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained YOLO model
    model = YOLO(model_path)

    # Supported image formats
    image_extensions = {".jpg", ".jpeg", ".png"}

    image_paths = [
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in image_extensions
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in: {input_dir}"
        )

    print(f"Found {len(image_paths)} test images.")

    for image_path in image_paths:
        # Load image using OpenCV
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Run YOLO object detection
        results = model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False
        )

        detection_count = 0

        # Draw detection results using OpenCV
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                class_id = int(box.cls[0])
                label = result.names[class_id]
                confidence = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw label and confidence
                text = f"{label} {confidence:.2f}"

                cv2.putText(
                    image,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                detection_count += 1

        # Save each result with its original filename
        output_path = output_dir / f"detected_{image_path.name}"

        cv2.imwrite(str(output_path), image)

        print(
            f"{image_path.name}: "
            f"{detection_count} objects detected -> "
            f"{output_path.name}"
        )

    print("OpenCV visualization completed.")


if __name__ == "__main__":
    predict_images()