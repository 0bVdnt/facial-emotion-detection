# Facial Emotion Detection using CNN(MobileNet)

## Overview

This project implements a **real-time facial emotion recognition system** using **transfer learning with MobileNet**. The model is trained on a cleaned version of the **FER2013 dataset** and classifies facial expressions into four emotions:

*  Fear
*  Happy
*  Neutral
*  Angry

The system also includes a **real-time webcam-based emotion detection pipeline** using OpenCV.

---

##  Features

*  Transfer learning using **MobileNet (ImageNet weights)**
*  Custom classification head for emotion detection
*  Data augmentation for improved generalization
*  Early stopping and learning rate scheduling
*  Model evaluation with:

  * Accuracy/Loss plots
  * Confusion matrix
  * Classification report
*  Real-time emotion detection via webcam

---

##  Model Architecture

* Base Model: **MobileNet (pretrained on ImageNet)**

* Top Layers:

  * Global Max Pooling
  * Dense (Softmax classifier)

* Partial fine-tuning:

  * First 15 layers frozen
  * Remaining layers trainable

---

##  Dataset

* Dataset: **FER2013 (cleaned version)**
* Input shape: **48 × 48 × 3**
* Classes used:

  * Fear
  * Happy
  * Neutral
  * Angry

---

##  Installation

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn scikit-plot
```

---

## Training Details

* Optimizer: **Adam**
* Loss: **Categorical Crossentropy**
* Batch Size: `25`
* Epochs: `40`

### Data Augmentation:

* Rotation
* Zoom
* Shear
* Width/Height shift
* Horizontal flip

---

##  Training Pipeline

1. Load dataset from directory structure
2. Convert images into NumPy arrays
3. Normalize pixel values (`/255`)
4. One-hot encode labels
5. Train-test split (90/10)
6. Apply data augmentation
7. Train MobileNet-based model
8. Save trained model (`.h5`)

---

##  Evaluation

* Confusion Matrix
* Classification Report
* Accuracy/Loss curves

###  Observations

* Model performs best on **Happy**
* Lower performance on:

  * Fear
  * Angry
* Likely causes:

  * Class imbalance
  * Ambiguous facial expressions
  * Dataset noise

---

## Real-Time Emotion Detection

Uses:

* **OpenCV Haar Cascade** for face detection
* Trained MobileNet model for classification

### How it works:

1. Capture webcam feed
2. Detect faces
3. Resize face to `224x224`
4. Preprocess using MobileNet preprocessing
5. Predict emotion
6. Display bounding box + label

### Run:

```bash
python emotion_realtime.py
```

Press **`q`** to quit.

---

##  Output Files

* `model_moblenet.h5` → trained model
* `model_mobelnet.yaml` → model architecture
* `mobilenet.png` → architecture visualization
* `epoch_history_mobilenet.png` → training plots
* `confusion_matrix_mobilenet.png` → confusion matrix

---

##  Known Issues

* Bias toward predicting **"Angry"** or dominant class
* Dataset contains mislabeled samples
* Limited emotion categories (only 4)

---

##  Future Improvements

* Use full FER2013 classes (7 emotions)
* Try deeper architectures (ResNet, EfficientNet)
* Apply class weighting or focal loss
* Improve preprocessing (face alignment)
* Deploy as web/mobile app

---

# Group Project by:
- Balaji Mukkawar   - RBT23CB022
- Sanjal Nale       - RBT23CB023
- Vedant Neve       - RBT23CB024
- Krushnakant Patil - RBT23CB025
