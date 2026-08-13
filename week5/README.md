# Text Generation using Vanilla RNN, LSTM, and GRU

## Overview

This project demonstrates text generation using three popular Recurrent Neural Network (RNN) architectures:

- Vanilla RNN
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The models are trained on a text corpus to learn grammar, sentence structure, and contextual relationships. After training, each model generates text by predicting the next word based on a given seed sequence.

---

## Objectives

- Understand sequential deep learning models.
- Implement Vanilla RNN, LSTM, and GRU for text generation.
- Compare the learning capabilities of different recurrent architectures.
- Analyze training loss and generated text quality.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Workflow

1. Import required libraries
2. Load and preprocess the text corpus
3. Tokenize the text
4. Create input sequences
5. Apply sequence padding
6. Build and train a Vanilla RNN model
7. Build and train an LSTM model
8. Build and train a GRU model
9. Compare training loss
10. Generate text using all three models
11. Analyze model performance

---

## Models Implemented

### 1. Vanilla RNN
A basic recurrent neural network that processes sequential data by maintaining a hidden state. It serves as the baseline model but struggles with long-term dependencies due to the vanishing gradient problem.

### 2. LSTM
Long Short-Term Memory (LSTM) networks use memory cells and gating mechanisms to retain important information over longer sequences, making them effective for language modeling.

### 3. GRU
Gated Recurrent Unit (GRU) simplifies the LSTM architecture by reducing the number of gates while maintaining comparable performance with fewer parameters.

---

## Results

The three models were evaluated by comparing:

- Training Loss
- Learning Stability
- Generated Text Quality

Observations indicate that:

- Vanilla RNN learns sequential patterns but has difficulty capturing long-range dependencies.
- LSTM generates more coherent and context-aware text.
- GRU achieves performance similar to LSTM while requiring fewer parameters and training faster.

---

## Project Structure

```
Week5_Text_Generation/
│── Week5_AmanNegi.ipynb
│── README.md
└── requirements.txt
```

---

## Learning Outcomes

- Learned text preprocessing for deep learning.
- Understood tokenization and sequence generation.
- Implemented three recurrent neural network architectures.
- Compared sequence modeling capabilities of RNN, LSTM, and GRU.
- Generated text using trained deep learning models.

---

## Conclusion

This project demonstrates the application of recurrent neural networks for text generation. While Vanilla RNN provides a simple baseline, LSTM and GRU significantly improve sequence learning through gating mechanisms, producing more coherent and contextually meaningful text. The comparison highlights the strengths and limitations of each architecture in natural language processing tasks.

---

## Author

**Aman Negi**

B.Tech Computer Science Engineering

DIT University

Celebal Technologies – Data Science Internship (Week 5)