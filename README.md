# CNN Image Classification — Mongolian License Plate Length

## Overview

This project focuses on classifying Mongolian vehicle license plates based on the number of characters present in the plate.

The task is a **binary classification problem**:

* Class 0 → Plate contains **6 characters**
* Class 1 → Plate contains **7 characters**

The model is implemented using **PyTorch** with a Convolutional Neural Network (CNN) architecture.

---

## Problem Statement

Given an image of a vehicle license plate, the goal is to predict whether the plate contains **6 or 7 total characters**.

This requires the model to learn visual patterns from plate structures, including variations in formatting and symbols.

---

## Dataset

This project uses a **synthetic dataset** based on Mongolian vehicle license plates.

* Total images: **3000**
* Training set: **1000 images**
* Test set: **2000 images**
* Image type: JPG

### Classes

* **0** → Plate contains 6 characters
* **1** → Plate contains 7 characters

### Notes

* Some plates include additional symbols such as **Soyombo**
* Country identifiers like **MNG / MGL** may appear

Dataset format includes:

* `train.csv` → training labels
* `sample_submission.csv` → submission format

---

## Model

A Convolutional Neural Network (CNN) is used to extract features from images and perform classification.

### Workflow

1. Load dataset from CSV
2. Preprocess images (resize, normalization)
3. Train CNN model
4. Evaluate performance using accuracy
5. Predict on unseen test data
6. Generate submission file

---

## Tech Stack

* Python
* PyTorch
* NumPy
* Pandas
* PIL (Image processing)

---

## Evaluation

The model is evaluated using **Accuracy score**.

### Submission Format

```
ID,label
0,0
1,1
2,0
```

---

## How to Run

```bash
# Clone repository
git clone https://github.com/BeBecpp/CNN-Image-Classification.git

# Navigate to project
cd CNN-Image-Classification

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Generate submission
python predict.py
```

---

## Results

The model is able to learn meaningful features from plate images and generalize to unseen data.

*(Add accuracy here if available, e.g. Accuracy: 0.87)*

---

## Future Improvements

* Improve model accuracy with hyperparameter tuning
* Add data augmentation
* Try deeper architectures (ResNet, EfficientNet)
* Build a web interface for image upload and prediction

---

## Author

Bayarbayasgalan (BeBe)

GitHub: [https://github.com/BeBecpp](https://github.com/BeBecpp)
Portfolio: [https://nero404.blogspot.com/](https://nero404.blogspot.com/)

---

## Note

This project is based on a competition-style dataset and demonstrates practical experience in:

* Deep learning workflows
* Image classification
* Data handling and preprocessing
* Model training and evaluation
<img width="280" height="720" alt="image" src="https://github.com/user-attachments/assets/af258ea5-caf8-4afc-a150-36f622cc09f7" />
