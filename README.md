# 🌸 Flower Object Detection using YOLO

A computer vision project that detects and classifies multiple flower species using YOLO.

This project trains a YOLO-based object detection model on a flower dataset containing 15 flower classes and performs object detection on unseen flower images.

The project was designed with a structured training and inference pipeline, configurable parameters, and unit tests for maintainability and reproducibility.

---

# 🎯 Project Objectives

- Build a flower object detection model using YOLO
- Detect and classify multiple flower species in images
- Train the model using GPU acceleration in Google Colab
- Evaluate object detection performance using Precision, Recall, and mAP
- Separate training parameters using a configuration file
- Build a reusable training and inference pipeline
- Verify project configurations using Unit Tests

---

# 🌷 Flower Classes

The dataset contains 15 flower classes:

- Carnation
- Chrysanthemum
- Daffodil
- Daisy
- Hydrangea
- Iris
- Jasmine
- Lotus
- Orchid
- Peony
- Primrose
- Rose
- Sunflower
- Tulip
- Zinna

---

# 📂 Project Structure

```text
flower-object-detection/
│
├── config/
│   └── config.yaml
│
├── data/
│   └── data.yaml
│
├── results/
│   ├── flower_detection/
│   └── predictions/
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_config.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Directory and File Description

### `config/`

Stores project configuration files.

- `config.yaml`
  - YOLO model configuration
  - Dataset path and number of classes
  - Training parameters such as epochs, image size, and batch size
  - Prediction confidence threshold
  - Output directory settings

Using a separate configuration file allows training parameters to be changed without modifying the Python source code.

### `data/`

Stores dataset configuration and local test images.

The full training dataset is not uploaded to GitHub due to its size.

### `results/`

Stores model training and prediction outputs.

Examples include:

- Training performance graphs
- Confusion Matrix
- Validation results
- Object detection result images

Large model weight files such as `.pt` files are excluded from Git tracking.

### `src/`

Contains the main Python source code.

- `train.py`
  - Loads training parameters from `config.yaml`
  - Loads the pretrained YOLO model
  - Trains the flower object detection model

- `predict.py`
  - Loads the trained YOLO model
  - Performs object detection on new flower images
  - Saves prediction results

### `tests/`

Contains Unit Tests using pytest.

- `test_config.py`
  - Verifies that `config.yaml` exists
  - Validates model configuration
  - Validates dataset configuration
  - Checks training parameter values
  - Validates prediction confidence threshold

---

# 🛠 Tech Stack

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- PyYAML
- pytest
- Google Colab GPU
- Roboflow

---

# 📊 Dataset

A flower Object Detection dataset from Roboflow Universe was used.

- Number of classes: 15
- Annotation format: YOLO
- License: CC BY 4.0

The dataset contains Bounding Box annotations for multiple flower species.

The dataset is divided into:

```text
train/
valid/
test/
```

The training dataset is downloaded directly in the Google Colab environment and is not included in this repository.

---

# ⚙️ Configuration

Training and prediction parameters are managed through:

```text
config/config.yaml
```

Example:

```yaml
model:
  name: "yolo11n.pt"

dataset:
  data_yaml: "data/data.yaml"
  num_classes: 15

training:
  epochs: 30
  image_size: 640
  batch_size: 16
  device: 0

prediction:
  confidence_threshold: 0.25

output:
  project_dir: "results"
  experiment_name: "flower_detection"
```

This structure allows model settings to be modified without changing the training or prediction source code.

---

# 🚀 Training Pipeline

The model training pipeline follows these steps:

```text
Flower Dataset
      ↓
YOLO Annotation
      ↓
Dataset Configuration (data.yaml)
      ↓
Project Configuration (config.yaml)
      ↓
Pretrained YOLO Model
      ↓
GPU Training
      ↓
Model Evaluation
      ↓
Best Model (best.pt)
      ↓
Flower Object Detection
```

The model is trained using a GPU environment in Google Colab.

---

# 🧪 Unit Test

Unit Tests are implemented using pytest.

Run:

```bash
py -m pytest
```

Current test result:

```text
5 passed
```

The tests verify the configuration required for the training and inference pipeline.

---

# 📈 Model Evaluation

The trained model will be evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion Matrix

Training and evaluation results will be added after model training is completed.

---

# 🖼 Detection Results

Object detection results will be added after inference using the trained model.

The final results will include:

- Original flower image
- YOLO Bounding Box detection result
- Predicted flower class
- Confidence score

---

# 💡 Mentor Feedback Applied

The following improvements were applied based on previous mentor feedback:

- Organized the project using a clear directory structure
- Added detailed descriptions of each directory and file to the README
- Separated model and training parameters using `config.yaml`
- Designed the project so that parameters can be modified without directly editing the training source code
- Used Google Colab GPU instead of local CPU for model training
- Maintained a Git Branch-based development workflow
- Added Unit Tests to verify project configuration

---

# 🔗 Connection to Previous Project

This project explores flower image analysis using Object Detection.

While the previous graduation project focuses on AI-based bouquet image generation, this project approaches the flower domain from an Object Detection perspective.

In the future, the object detection model could potentially be used to analyze generated bouquet images and verify whether specific flower types are present in the generated result.

---

# 🔮 Future Improvements

- Analyze Precision, Recall, and mAP for each flower class
- Analyze false positive and false negative detection cases
- Improve detection performance through additional training and hyperparameter tuning
- Compare different YOLO model sizes
- Test the model on real-world bouquet images
- Explore detection of multiple flower species within a single bouquet
- Integrate flower detection with an AI-based bouquet generation pipeline