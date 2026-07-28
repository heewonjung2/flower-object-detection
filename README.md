# Flower Object Detection

YOLO 기반 꽃 객체 탐지 모델을 학습하고, OpenCV를 활용하여 실제 이미지에서 꽃 종류를 탐지하고 결과를 시각화하는 프로젝트입니다.

## Dataset

Roboflow Universe의 Flower Object Detection Dataset을 활용했습니다.

- Task: Object Detection
- Classes: 15
- Classes:
  - carnation
  - chrysanthemum
  - daffodil
  - daisy
  - hydrangea
  - iris
  - jasmine
  - lotus
  - orchid
  - peony
  - primrose
  - rose
  - sunflower
  - tulip
  - zinna

## Model Training

Ultralytics YOLO11n 모델을 사용하여 Google Colab의 Tesla T4 GPU 환경에서 30 Epoch 학습을 진행했습니다.

### Validation Results

| Metric | Result |
| --- | ---: |
| Precision | 0.777 |
| Recall | 0.765 |
| mAP50 | 0.809 |
| mAP50-95 | 0.561 |

전체 클래스 기준 mAP50은 0.809를 기록했습니다.

클래스별 성능을 확인한 결과 iris, hydrangea, lotus 등의 클래스는 높은 탐지 성능을 보였지만, peony, tulip 등 일부 클래스에서는 상대적으로 낮은 성능을 확인했습니다.

## OpenCV Inference

학습된 YOLO 모델의 `best.pt`를 로컬 환경에서 불러와 여러 테스트 이미지에 대해 객체 탐지를 수행했습니다.

OpenCV를 활용하여 다음 정보를 이미지 위에 시각화했습니다.

- Bounding Box
- 예측 클래스
- Confidence Score

`data/test_images/` 폴더의 여러 이미지를 일괄 처리하고, 탐지 결과를 `results/opencv_predictions/`에 자동 저장하도록 구현했습니다.

## Test Results

다양한 환경의 이미지를 사용하여 모델의 실제 탐지 성능을 확인했습니다.

- 단일 꽃 이미지에서는 iris를 높은 confidence로 정확하게 탐지했습니다.
- 복잡한 꽃다발 이미지에서는 일부 꽃이 탐지되지 않는 미탐(False Negative)이 발생했습니다.
- 여러 종류의 꽃이 섞인 꽃다발에서는 서로 다른 꽃을 동일한 클래스로 분류하는 오탐(False Positive)이 발생했습니다.
- 연꽃 이미지에서는 실제 연꽃을 높은 confidence로 탐지했지만, 사람 얼굴을 lotus로 잘못 탐지하는 사례가 확인되었습니다.
- 학습 데이터에 포함되지 않은 lily 이미지에서는 chrysanthemum으로 잘못 분류하는 OOD(Out-of-Distribution) 문제가 확인되었습니다.

## Confidence Threshold Experiment

초기 confidence threshold를 0.25로 설정하여 테스트한 결과, 낮은 confidence의 오탐이 다수 발생했습니다.

이를 개선하기 위해 threshold를 0.5로 변경하여 동일한 이미지에 대해 다시 추론했습니다.

### Threshold 0.25

- 다양한 객체를 탐지하지만 낮은 confidence의 오탐이 발생
- 꽃이 아닌 영역 또는 다른 종류의 꽃을 잘못 탐지하는 사례 확인

### Threshold 0.5

- 낮은 confidence의 오탐 감소
- 높은 confidence를 가진 주요 객체 중심으로 결과가 정리됨
- 일부 이미지에서는 탐지 결과가 사라져 미탐 가능성이 증가함
- 사람 얼굴을 lotus로 탐지하는 고신뢰도 오탐은 여전히 존재

따라서 본 프로젝트에서는 오탐 감소를 위해 최종 confidence threshold를 **0.5**로 설정했습니다.

## Limitations

실험을 통해 다음과 같은 한계점을 확인했습니다.

1. 학습 데이터와 실제 꽃다발 이미지 사이의 도메인 차이
2. 여러 종류의 꽃이 겹쳐 있는 환경에서 클래스 간 혼동 발생
3. 복잡한 배경과 작은 객체에 대한 탐지 성능 저하
4. 사람 등 비꽃 객체를 꽃으로 탐지하는 False Positive 발생
5. 학습하지 않은 꽃 종류를 기존 15개 클래스 중 하나로 잘못 분류하는 OOD 문제

## Future Improvements

향후 다음과 같은 방법으로 모델 성능을 개선할 수 있습니다.

- 실제 꽃다발 이미지 중심의 학습 데이터 추가
- 사람, 실내 배경 등 다양한 Negative Sample 추가
- 클래스별 데이터 수 불균형 개선
- 데이터 증강을 통한 다양한 촬영 환경 학습
- Confidence Threshold 및 모델 파라미터 최적화
- OOD Detection을 적용하여 미학습 클래스에 대한 Unknown 처리
- 실제 서비스 환경에 적합한 꽃 객체 탐지 및 생성 이미지 검증 모델로 확장

## Project Structure

```text
flower-object-detection/
├── config/
│   └── config.yaml
├── data/
│   └── test_images/
├── results/
│   ├── flower_detection/
│   │   └── weights/
│   │       └── best.pt
│   └── opencv_predictions/
├── src/
│   ├── train.py
│   └── predict.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Run

### Install Dependencies

```bash
py -m pip install -r requirements.txt
```

### Run Prediction

```bash
py src/predict.py
```

### Run Tests

```bash
py -m pytest
```