# Autoencoder for Image Denoising using MNIST

## Overview

This project implements a **Convolutional Autoencoder** for removing noise from handwritten digit images using the **MNIST dataset**.

Artificial Gaussian noise is added to clean MNIST images, and the autoencoder is trained to reconstruct the original clean images from their noisy versions.

The project demonstrates the use of **autoencoders, representation learning, convolutional neural networks, and image reconstruction** for an image denoising task.

---

## Objective

The objective of this project is to build a deep learning model capable of:

- Learning meaningful representations of handwritten digit images
- Removing artificial noise from corrupted images
- Reconstructing images that closely resemble the original clean samples
- Evaluating denoising performance using visual comparison and reconstruction error

---

## Dataset

The project uses the **MNIST handwritten digit dataset**.

The dataset contains grayscale images of handwritten digits from **0 to 9**.

- Training Images: **60,000**
- Testing Images: **10,000**
- Image Size: **28 × 28**
- Image Type: Grayscale

---

## Project Workflow

1. Load the MNIST dataset
2. Normalize image pixel values
3. Add Gaussian noise to the images
4. Visualize clean and noisy images
5. Build a Convolutional Autoencoder
6. Train the model using noisy images as input and clean images as targets
7. Analyze training and validation loss
8. Generate denoised images
9. Compare noisy, denoised, and original images
10. Evaluate reconstruction performance using Mean Squared Error (MSE)

---

## Autoencoder Architecture

The model consists of two major components:

### Encoder

The encoder uses convolution and pooling layers to extract important visual features and compress the input image into a lower-dimensional representation.

### Decoder

The decoder uses convolution and upsampling layers to reconstruct the clean image from the compressed representation.

### Architecture Flow

```text
Noisy Image (28 × 28 × 1)
        ↓
     Encoder
        ↓
Compressed Representation (7 × 7 × 16)
        ↓
     Decoder
        ↓
Reconstructed Image (28 × 28 × 1)
```

The model contains **12,193 trainable parameters**.

---

## Model Training

The autoencoder was trained using:

- Optimizer: **Adam**
- Loss Function: **Binary Cross-Entropy**
- Epochs: **10**
- Batch Size: **128**

The noisy images were used as model inputs, while the original clean images were used as target outputs.

```text
Noisy Image → Autoencoder → Reconstructed Clean Image
```

---

## Training Results

After 10 epochs:

- Training Loss: **0.0889**
- Validation Loss: **0.0886**

The close training and validation losses indicate that the model generalized well to unseen test images.

---

## Denoising Evaluation

Mean Squared Error (MSE) was used to compare the noisy and reconstructed images with the original clean images.

| Image Type | MSE |
|---|---:|
| Noisy Images | 0.07964 |
| Denoised Images | 0.00910 |

The autoencoder achieved approximately an **88.6% reduction in reconstruction error** after denoising.

---

## Results

The trained autoencoder successfully removed a significant amount of artificial noise from MNIST images.

The reconstructed images:

- Preserved the overall structure of handwritten digits
- Removed most of the random noise
- Closely resembled the original clean images
- Generalized across different digit classes from 0 to 9

Some reconstructed images appeared slightly smoother than the originals, which is expected due to compression and reconstruction.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

```text
Week6/
│
├── week6_AmanNegi.ipynb
├── README.md
└── requirements.txt
```

The MNIST dataset can be downloaded separately and placed in the appropriate dataset directory.

---

## Key Learnings

Through this project, I gained practical experience with:

- Autoencoders
- Encoder and Decoder architectures
- Convolutional neural networks
- Image preprocessing and normalization
- Artificial noise generation
- Latent representations
- Image reconstruction
- Reconstruction loss
- Mean Squared Error evaluation
- Image denoising using deep learning

---

## Conclusion

This project demonstrates how a Convolutional Autoencoder can be used effectively for image denoising.

By training the model to reconstruct clean MNIST images from noisy inputs, the autoencoder learned meaningful visual representations of handwritten digits.

The reduction in MSE from **0.07964 to 0.00910**, along with the visual quality of the reconstructed images, demonstrates that the trained autoencoder successfully removed noise while preserving important image structures.