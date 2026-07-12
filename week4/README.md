# Week 4 – CIFAR-10 Image Classification using ANN & CNN

## Overview

This project focuses on building and comparing deep learning models for image classification using the CIFAR-10 dataset. Multiple neural network architectures and training strategies were implemented to analyze their performance on a multi-class image classification problem.

---

## Objective

- Build an Artificial Neural Network (ANN) for image classification.
- Build a Convolutional Neural Network (CNN).
- Compare the performance of ANN and CNN.
- Improve CNN performance using Data Augmentation.
- Analyze learning curves and model performance.

---

## Dataset

**CIFAR-10**

The CIFAR-10 dataset contains **60,000 color images** of size **32 × 32 pixels** belonging to **10 different classes**.

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset Split:
- Training Images: 50,000
- Test Images: 10,000

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas

---

## Workflow

1. Load the CIFAR-10 dataset
2. Visualize sample images
3. Preprocess and normalize image data
4. Build and train an ANN model
5. Build and train a CNN model
6. Apply Data Augmentation
7. Train an augmented CNN model
8. Compare model performance
9. Visualize learning curves
10. Analyze results and draw conclusions

---

## Models Implemented

### Artificial Neural Network (ANN)

- Fully Connected Dense Layers
- Dropout Regularization
- Adam Optimizer

---

### Convolutional Neural Network (CNN)

- Convolution Layers
- Max Pooling Layers
- Dense Layers
- Dropout
- Adam Optimizer

---

### CNN with Data Augmentation

Data augmentation techniques applied:

- Random Horizontal Flip
- Random Rotation
- Random Zoom

These transformations improve model robustness by generating more diverse training samples.

---

## Model Comparison

| Model | Test Accuracy |
|--------|--------------:|
| ANN | 44.23% |
| CNN | 71.36% |
| CNN + Data Augmentation | 67.91% |

---

## Key Observations

- ANN provides a baseline but struggles with image data because it ignores spatial relationships.
- CNN significantly outperforms ANN by automatically learning spatial features through convolution.
- Data augmentation improves model robustness and generalization, although it may require additional training epochs to achieve its best performance.
- CNN is the preferred architecture for image classification tasks.

---

## Skills Demonstrated

- Image Classification
- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Data Augmentation
- Deep Learning
- TensorFlow & Keras
- Model Evaluation
- Computer Vision Fundamentals

---

## Repository Structure

```
week4/
│── week4_AmanNegi.ipynb
│── README.md
│── requirements.txt
```

---

## Author

**Aman Negi**

B.Tech Computer Science Engineering  
DIT University

GitHub: https://github.com/amanpy45-png

---

## Note

This project was completed as part of the **Celebal Technologies Data Science Internship (Week 4)** to understand the fundamentals of deep learning and computer vision through image classification using the CIFAR-10 dataset.