from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_config_file_exists():
    """Check if config.yaml exists."""
    assert CONFIG_PATH.exists()


def test_model_config():
    """Check model configuration."""
    config = load_config()

    assert "model" in config
    assert config["model"]["name"] == "yolo11n.pt"


def test_dataset_config():
    """Check dataset configuration."""
    config = load_config()

    assert "dataset" in config
    assert config["dataset"]["num_classes"] == 15


def test_training_config():
    """Check training configuration."""
    config = load_config()

    training = config["training"]

    assert training["epochs"] > 0
    assert training["image_size"] > 0
    assert training["batch_size"] > 0


def test_prediction_config():
    """Check prediction configuration."""
    config = load_config()

    confidence = config["prediction"]["confidence_threshold"]

    assert 0 <= confidence <= 1