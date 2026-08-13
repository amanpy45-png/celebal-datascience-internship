# Visual Product Recommender

An image-based fashion product recommendation system that retrieves visually similar products using deep learning, transfer learning, and metric learning.

The system accepts a product image, extracts deep visual features, and retrieves the most visually similar products from a fashion product dataset.

It implements two approaches:

- **ResNet50 baseline** using 2048-dimensional embeddings
- **Siamese Network** using compact 128-dimensional embeddings learned with triplet loss

The project also provides an interactive Streamlit interface for image-based product search.

---

## Repository

GitHub: https://github.com/amanpy45-png/celebal-datascience-internship

---

## Problem Statement

In e-commerce platforms, users may want to find products that look similar to an item they already have.

Traditional keyword-based search cannot always capture visual characteristics such as:

- Color
- Shape
- Style
- Texture
- Design

This project addresses this problem by building an image-based recommendation engine that retrieves visually similar fashion products using deep learning embeddings.

---

## Objectives

- Accept a product image as input.
- Extract meaningful visual features using a pretrained CNN.
- Retrieve Top-K visually similar products.
- Build a ResNet50-based baseline recommendation system.
- Develop a Siamese Network using triplet loss.
- Compare both approaches using Precision@5 and retrieval time.
- Provide an interactive Streamlit interface for image-based search.

---

## System Architecture

```
                  Input Product Image
                          |
                          v
                 Image Preprocessing
                          |
                          v
               +----------------------+
               |  Feature Extraction  |
               +----------------------+
                    /            \
                   /              \
                  v                v
          ResNet50 Baseline   Siamese Network
          2048-D Embedding    128-D Embedding
                  |                |
                  +-------+--------+
                          |
                          v
                  Cosine Similarity
                          |
                          v
                    Top-K Products
                          |
                          v
                    Streamlit UI
```

---

## Methodology

### 1. Dataset Preparation

The project uses the Fashion Product Images dataset.

To make experimentation computationally practical, a subset of 2,000 images was created from 8 product categories:

| Category      | Images |
|---------------|--------|
| Tshirts       | 250    |
| Shirts        | 250    |
| Casual Shoes  | 250    |
| Sports Shoes  | 250    |
| Kurtas        | 250    |
| Tops          | 250    |
| Handbags      | 250    |
| Heels         | 250    |
| **Total**     | **2,000** |

The complete dataset is not included in this repository because of its large size.

### 2. Image Preprocessing

Images are:

- Resized to 224 × 224
- Converted to RGB
- Preprocessed using ImageNet preprocessing
- Prepared for ResNet50 feature extraction

### 3. ResNet50 Baseline

A pretrained ResNet50 model with ImageNet weights is used as the baseline feature extractor.

The classification head is removed and global average pooling is used to generate a **2048-dimensional image embedding**.

These embeddings represent the visual characteristics of each product.

Cosine similarity is then used to retrieve visually similar products.

### 4. Siamese Network

A Siamese-style embedding network was implemented using ResNet50 as the backbone.

The network:

- Uses pretrained ImageNet weights
- Freezes most ResNet50 layers
- Fine-tunes the last few layers
- Projects features into a 128-dimensional embedding space
- Applies L2 normalization
- Uses triplet loss for similarity learning

The triplet structure consists of:

```
                 Anchor
                /      \
               /        \
              v          v
        Positive        Negative
      Similar Item    Different Item
```

The objective is to make the anchor closer to the positive example and farther from the negative example in the learned embedding space.

---

## Recommendation Pipeline

```
Upload Product Image
        |
        v
Resize + Preprocess
        |
        v
Generate Image Embedding
        |
        v
Compare with Product Embeddings
        |
        v
Calculate Cosine Similarity
        |
        v
Sort by Similarity
        |
        v
Return Top-K Recommendations
```

---

## Key Features

- Image-based product search
- No text query required
- Deep learning feature extraction
- ResNet50 transfer learning
- Siamese Network
- Triplet-loss-based metric learning
- Compact 128-dimensional embeddings
- Cosine similarity search
- Top-K recommendations
- Similarity scores
- Product category information
- Interactive Streamlit interface
- Model comparison and evaluation

---

## Model Evaluation

The models were evaluated using Precision@5 and retrieval time.

| Model             | Embedding Dimension | Precision@5 | Retrieval Time |
|-------------------|----------------------|-------------|-----------------|
| ResNet50 Baseline | 2048                 | 86.8%       | 0.0242 sec      |
| Siamese Network   | 128                  | 84.8%       | 0.00116 sec     |

### Results

The ResNet50 baseline achieved slightly higher Precision@5 in this experiment.

However, the Siamese Network reduced the embedding size from 2048 dimensions → 128 dimensions and achieved significantly faster similarity retrieval once the embeddings were available.

This demonstrates a trade-off between retrieval accuracy, embedding size, and retrieval speed.

---

## Example

Given an input image of a red T-shirt, the system retrieves visually similar products from the dataset.

**Example output — Top 5 Similar Products:**

1. Tshirts — Similarity: 0.776
2. Tshirts — Similarity: 0.759
3. Tshirts — Similarity: 0.739
4. Tshirts — Similarity: 0.734
5. Tops — Similarity: 0.731

The Streamlit interface displays the query image together with the recommended products, categories, and similarity scores.

---

## Project Structure

```
Visual_Product_Recommender/
│
├── app.py
├── build_embeddings.py
├── evaluate_models.py
├── test_recommendation.py
├── train_siamese.py
├── requirements.txt
├── README.md
│
├── data/
│   └── subset.csv
│
├── embeddings/
│   ├── baseline_embeddings.npy
│   └── metadata.csv
│
├── models/
│   └── siamese_embedding.keras
│
└── utils/
    ├── dataset.py
    ├── evaluation.py
    ├── feature_extractor.py
    ├── siamese.py
    └── similarity.py
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/amanpy45-png/celebal-datascience-internship.git
cd celebal-datascience-internship/Visual_Product_Recommender
```

### 2. Create a Virtual Environment

```bash
python -m venv myenv
```

Activate it on Windows:

```bash
myenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The complete Fashion Product Images dataset is not included in this repository because of its large size.

After downloading the dataset, place it inside:

```
data/
└── fashion-dataset/
    ├── images/
    ├── images.csv
    └── styles.csv
```

The project uses a 2,000-image subset for experimentation.

To create the subset:

```bash
python utils/dataset.py
```

This generates:

```
data/subset.csv
```

---

## Generate Embeddings

Generate the baseline ResNet50 embeddings using:

```bash
python build_embeddings.py
```

This generates:

```
embeddings/
├── baseline_embeddings.npy
└── metadata.csv
```

---

## Train the Siamese Network

Train the Siamese embedding model using:

```bash
python train_siamese.py
```

The trained model is saved as:

```
models/siamese_embedding.keras
```

---

## Run Evaluation

Compare the ResNet50 baseline and Siamese Network:

```bash
python evaluate_models.py
```

The evaluation reports:

- Precision@5
- Retrieval time
- Embedding representation

---

## Test Recommendation

Run a command-line recommendation test:

```bash
python test_recommendation.py
```

---

## Run the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

The application provides:

- Product image upload
- Model selection
- Number of recommendations
- Similar product visualization
- Similarity scores
- Product categories

---

## Technologies Used

- Python
- TensorFlow
- Keras
- ResNet50
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Pillow

---

## Limitations

- The experiment uses a subset of the complete dataset.
- Recommendations are based on visual similarity rather than price, brand, or user preferences.
- Siamese Network performance depends on the quality of generated training triplets.
- Embedding generation can be computationally expensive on CPU.
- The current system is a visual similarity search engine rather than a fully personalized recommendation system.

---

## Future Improvements

- Use FAISS for scalable similarity search.
- Train on a larger dataset.
- Improve triplet sampling strategies.
- Experiment with EfficientNet and other pretrained architectures.
- Incorporate product metadata and text information.
- Implement hybrid visual + textual recommendation.
- Add personalized recommendations using user interaction history.
- Deploy the application using a cloud platform.

---

## Conclusion

This project demonstrates how deep learning, transfer learning, and metric learning can be used to build an image-based visual product recommendation system.

The ResNet50 baseline provides strong visual retrieval performance, while the Siamese Network learns a compact embedding representation that enables faster similarity retrieval.

The project combines:

**Deep Learning + Transfer Learning + Metric Learning + Similarity Search + Streamlit**

into an end-to-end visual product recommendation pipeline.
