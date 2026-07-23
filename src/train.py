from pathlib import Path

import yaml
from ultralytics import YOLO


# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent

# config.yaml 경로
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config():
    """Load project configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_model():
    """Train YOLO flower object detection model."""
    config = load_config()

    model_config = config["model"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    output_config = config["output"]

    # Load pretrained YOLO model
    model = YOLO(model_config["name"])

    # Train model
    model.train(
        data=dataset_config["data_yaml"],
        epochs=training_config["epochs"],
        imgsz=training_config["image_size"],
        batch=training_config["batch_size"],
        device=training_config["device"],
        project=output_config["project_dir"],
        name=output_config["experiment_name"],
    )


if __name__ == "__main__":
    train_model()