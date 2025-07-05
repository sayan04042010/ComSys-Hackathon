# ComSys-Hackathon
TaskA
## Gender Classification using EfficientNetB3 

This project is a deep learning-based gender classification system built on top of *EfficientNetB3*, fine-tuned for binary classification (Male/Female). It includes preprocessing with Albumentations, Mediapipe face cropping, training with PyTorch, and experiment tracking using Weights & Biases (optional).

---

## Features

- Face detection using **Mediapipe**
- EfficientNetB3 pretrained on ImageNet, fine-tuned for gender
- Metrics: Accuracy, Precision, Recall, F1-Score
- Real-time training progress with **tqdm**
- Supports GPU acceleration on Google Colab
- Evaluation on validation data
- Plots and metrics for comparison and analysis

---

## Folder Structure

```
├── dataset/
│   ├── train/
│   ├── val/
├── models/
│   └── efficientnet_b3_gender.pth
├── utils/
│   └── crop_face.py
├── main.py
├── train.py
├── eval.py
├── requirements.txt
```

---

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- albumentations
- mediapipe
- scikit-learn
- tqdm
- wandb *(optional)*

Install with:

```bash
pip install -r requirements.txt
```

---

## Data Preprocessing

- Uses `Mediapipe` for face detection and cropping
- Albumentations transforms:
  - Resize, Normalize
  - Random Crop, Flip, Blur (for training set)

---

## Model Architecture

- Base: `EfficientNetB3` from `timm` library
- Final layer replaced with: `nn.Linear(in_features, 1)`
- Binary classification with `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Scheduler: `StepLR` (optional)

---

## Training

```bash
python train.py
```

- Tracks training/validation accuracy
- Supports early stopping and model checkpointing

---

## Evaluation

After training:

```bash
python eval.py
```

Prints:

- Validation Accuracy
- Precision
- Recall
- F1-Score

---

## Example Results

```
Training Accuracy:     90%
Validation Accuracy:   87.91%
Precision:             90.56%
Recall:                95.04%
F1-Score:              92.75%
```

No overfitting detected — healthy validation scores.

---

## Future Improvements

- Add age prediction as multi-task learning
- Try ViT or ConvNext architectures
- Deploy as a real-time web or mobile app

---

## Author

**Saheli Deb, Sayan Chatterjee, Indranil Mukhopadhyay**

---

## License

This project is open-source under the MIT License. Feel free to use and modify.

Task B

# Task_B: Face Recognition from Distorted Images using DeepFace

This project evaluates a face recognition system that can correctly identify people from **distorted or degraded images** (e.g., blur, fog, rain). It uses **DeepFace** with the **ArcFace** model and a **K-Nearest Neighbors (KNN)** classifier to embed, compare, and classify images based on their face embeddings.

---

## Folder Structure

```
Task_B/
├── train/
│   ├── train/
│   │   ├── 001_frontal/
│   │   │    ├── 001_frontal.jpg
│   │   │    └── distortion/
│   │   │         ├── blur.jpg
│   │   │         ├── fog.jpg
│   │   │         └── ...
│   │   ├── 002_frontal/
│   │   │    └── ...
│   │   └── ...
├── val/
│   ├── val/
│   │   ├── 009_frontal/
│   │   │    ├── 009_frontal.jpg
│   │   │    └── distortion/
│   │   ├── 017_frontal/
│   │   └── ...
```

---

## Objective

To build a **robust face recognition system** that can recognize distorted face images by matching them with clean training images using embedding vectors.

---

## How It Works

1. Upload `train.zip` and `val.zip`
2. Extract into: `Task_B/train` and `Task_B/val`
3. Load reference images (clear ones) and distorted validation images
4. Extract **512-d embeddings** using **DeepFace with ArcFace**
5. Fit a **KNN classifier** (k=1) on training embeddings
6. Predict labels of distorted validation faces
7. Evaluate performance with accuracy, F1, etc.

---

## Features

- Handles **blur, fog, low-light, and distorted faces**
- Uses `enforce_detection=False` to skip unreadable faces
- Prints warnings if no face is detected
- Deeply nested folder support (train/train/ and val/val/)
- Metrics for quantitative evaluation

---

## Requirements

- Google Colab or Python 3.7+
- Install dependencies:

```bash
pip install deepface opencv-python scikit-learn tqdm
```

---

## Evaluation Metrics

- Top-1 Accuracy
- Macro F1-Score
- Macro Precision
- Macro Recall

These help measure how well the model performs on distorted images.

---

## Known Issues

- Some distorted images may not contain detectable faces → skipped
- Embedding extraction is **slow** if dataset has 10,000+ images
- Assumes 1 clear image + many distortions per person

---

## Author

- **Saheli Deb, Sayan Chatterjee, Indranil Mukhopadhyay**
- Project: Task_B | Face Recognition on Noisy Inputs
- Date: July 2025

---

## License

This project is licensed under the **MIT License**.
