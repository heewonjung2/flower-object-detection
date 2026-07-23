from pathlib import Path

import yaml
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config():
    """Load project configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def predict_image(image_path):
    """Detect flower objects in an input image."""
    config = load_config()

    confidence = config["prediction"]["confidence_threshold"]

    # Load the trained YOLO model
    model_path = BASE_DIR / "results" / "flower_detection" / "weights" / "best.pt"
    model = YOLO(model_path)

    # Run object detection
    results = model.predict(
        source=image_path,
        conf=confidence,
        save=True,
        project=BASE_DIR / "results",
        name="predictions",
    )

    return results


if __name__ == "__main__":
    predict_image("data/test.jpg")