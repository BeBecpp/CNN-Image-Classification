# 🧠 CNN Image Classification 

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** using PyTorch for image classification.
The model is trained and evaluated on a dataset provided via Kaggle competition.

The goal is to build a simple but effective deep learning pipeline for classifying images into predefined categories.

---

## 🚀 Features

* Custom CNN architecture built with PyTorch
* Custom Dataset & DataLoader implementation
* Image preprocessing using torchvision transforms
* Train/Test split with sklearn
* Model evaluation using accuracy score
* Ready for Kaggle submission workflow

---

## 🧱 Model Architecture

The model (`SimpleCNN`) consists of:

* Convolutional layers:

  * Conv2D → ReLU → MaxPool
  * Conv2D → ReLU → MaxPool
* Fully connected layers:

  * Flatten
  * Linear → ReLU → Dropout
  * Output layer

---

## 📂 Dataset

* Source: Kaggle Competition
* Structure:

  * `train/` – training images
  * `test/` – test images
  * `train.csv` – labels
  * `sample_submission.csv`

---

## ⚙️ Tech Stack

* Python
* PyTorch
* torchvision
* NumPy
* scikit-learn

---

## 🛠 Installation

```bash
pip install torch torchvision numpy scikit-learn
```

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Run training:

```bash
python train.py
```

---

## 📊 Evaluation

* Metric: **Accuracy**
* Uses `sklearn.metrics.accuracy_score`

---

## 📈 Future Improvements

* Add data augmentation
* Use pretrained models (ResNet, EfficientNet)
* Add confusion matrix & visualization
* Hyperparameter tuning
* Deploy as API (FastAPI)

---

## 🌐 Portfolio Extension (Recommended)

This project can be extended by:

* Creating a REST API for predictions
* Building a frontend (React/Vue)
* Upload image → get prediction

---

## 👨‍💻 Author

**Bayarbayasgalan (BeBe)**
Frontend Developer | Exploring AI & Machine Learning

---
